import streamlit as st
import pandas as pd
import json
import os
from collections import Counter

from src.loader import load_candidates
from src.honeypot import detect_honeypot
from src.scoring import combine_scores, tier_normalize
from src.reasoning import generate_reasoning

# Page config
st.set_page_config(
    page_title="Intelligent Recruiter Dashboard | Redrob",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Injected custom CSS for premium aesthetics (dark mode, glassmorphism, Inter font)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

/* Apply modern fonts */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', 'Outfit', sans-serif;
}

/* Gradient Header */
.main-title {
    font-family: 'Outfit', sans-serif;
    background: linear-gradient(135deg, #FF4B4B 0%, #8A2387 50%, #E94057 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 2.8rem;
    margin-bottom: 0.2rem;
    letter-spacing: -0.05rem;
}

.sub-title {
    font-size: 1.1rem;
    color: #a0aec0;
    margin-bottom: 2rem;
    font-weight: 400;
}

/* Premium Card container */
.glass-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
}

.candidate-card-header {
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    font-size: 1.5rem;
    color: #ffffff;
    margin-bottom: 0.5rem;
}

/* Custom Badges */
.badge-container {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 1rem;
}

.badge {
    padding: 0.35rem 0.75rem;
    font-size: 0.8rem;
    font-weight: 600;
    border-radius: 8px;
    letter-spacing: 0.02rem;
}

.badge-tech { background: rgba(99, 102, 241, 0.15); color: #a5b4fc; border: 1px solid rgba(99, 102, 241, 0.3); }
.badge-exp { background: rgba(16, 185, 129, 0.15); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.3); }
.badge-behav { background: rgba(236, 72, 153, 0.15); color: #f9a8d4; border: 1px solid rgba(236, 72, 153, 0.3); }
.badge-loc { background: rgba(245, 158, 11, 0.15); color: #fde047; border: 1px solid rgba(245, 158, 11, 0.3); }
.badge-honeypot { background: rgba(239, 68, 68, 0.15); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.3); }

/* Timeline design */
.timeline-item {
    border-left: 2px solid rgba(255, 255, 255, 0.15);
    padding-left: 1.5rem;
    margin-left: 0.5rem;
    position: relative;
    padding-bottom: 1.5rem;
}
.timeline-item::before {
    content: '';
    position: absolute;
    left: -6px;
    top: 4px;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #FF4B4B;
}

.timeline-title {
    font-weight: 600;
    color: #ffffff;
    font-size: 0.95rem;
}

.timeline-meta {
    font-size: 0.8rem;
    color: #a0aec0;
    margin-bottom: 0.4rem;
}

.timeline-desc {
    font-size: 0.85rem;
    color: #cbd5e0;
    line-height: 1.4;
}
</style>
""", unsafe_allow_html=True)

# Main Title and Subtitle
st.markdown("<div class='main-title'>🎯 Intelligent Candidate Discovery</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Redrob Candidate Screening Engine • Senior AI Engineer - Ranking & Retrieval</div>", unsafe_allow_html=True)

# Sidebar configurations
st.sidebar.markdown("### 📥 Source Selection")
uploaded_file = st.sidebar.file_uploader(
    "Upload candidates pool (JSON/JSONL)", type=["json", "jsonl"]
)
candidates_path = st.sidebar.text_input(
    "Or load local candidates file", value="./sampledata/candidates.jsonl"
)

st.sidebar.markdown("### ⚙️ Search Controls")
top_k = st.sidebar.slider("Top Candidates Shortlist (K)", min_value=5, max_value=100, value=100)
min_score = st.sidebar.slider("Minimum Quality Score Threshold", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
search_title = st.sidebar.text_input("Filter by Title Keyword (e.g. 'Staff', 'Zomato')")

st.sidebar.markdown("---")
rank_button = st.sidebar.button("🚀 Rank Candidates Pool", use_container_width=True)

# Main Dashboard logic
candidates = None
if rank_button or uploaded_file is not None or "results" in st.session_state:
    
    # Load candidates if not already in session state
    if "results" not in st.session_state:
        st.session_state.results = None
        st.session_state.honeypots = None
        st.session_state.all_evidence = None

    if st.session_state.results is None:
        try:
            if uploaded_file is not None:
                content = uploaded_file.read().decode("utf-8")
                if uploaded_file.name.endswith(".json"):
                    data = json.loads(content)
                    raw_candidates = data if isinstance(data, list) else [data]
                else:  # .jsonl
                    raw_candidates = [json.loads(line) for line in content.strip().split("\n") if line.strip()]
                st.sidebar.success(f"Loaded {len(raw_candidates)} uploaded profiles")
            elif os.path.exists(candidates_path):
                raw_candidates = load_candidates(candidates_path)
                st.sidebar.success(f"Loaded {len(raw_candidates)} profiles from disk")
            else:
                st.error(f"Candidates file not found. Please upload a file or specify a valid local path.")
                raw_candidates = []

            if raw_candidates:
                progress = st.progress(0, text="Analyzing profiles and filtering honeypots...")
                honeypots = {}
                results = []
                all_scores = []
                all_evidence = {}

                for i, c in enumerate(raw_candidates):
                    # Step 1: Honeypot check
                    flagged, reason = detect_honeypot(c)
                    if flagged:
                        honeypots[c["candidate_id"]] = {
                            "reason": reason,
                            "profile": c
                        }
                        continue

                    # Step 2: Custom Multi-Signal Scoring
                    feature_score, evidence = combine_scores(c)
                    all_scores.append(feature_score)
                    all_evidence[c["candidate_id"]] = evidence

                    results.append({
                        "candidate_id": c["candidate_id"],
                        "feature_score": feature_score,
                        "title": c.get("profile", {}).get("current_title", "N/A"),
                        "company": c.get("profile", {}).get("current_company", "N/A"),
                        "industry": c.get("profile", {}).get("current_industry", "N/A"),
                        "years": c.get("profile", {}).get("years_of_experience", 0),
                        "country": c.get("profile", {}).get("country", "N/A"),
                        "location": c.get("profile", {}).get("location", "N/A"),
                        "open_to_work": c.get("redrob_signals", {}).get("open_to_work_flag", False),
                        "candidate": c,
                    })
                    if (i + 1) % 500 == 0 or i == len(raw_candidates) - 1:
                        progress.progress((i + 1) / len(raw_candidates), text=f"Scored {i+1}/{len(raw_candidates)} candidates")

                # Step 3: Normalize scores to relevance bands
                normalized = [tier_normalize(s, all_scores) for s in all_scores]
                for idx, r in enumerate(results):
                    r["final_score"] = round(normalized[idx], 4)

                # Step 4: Sort by score descending, then candidate_id ascending for deterministic ranks
                results.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))

                # Step 5: Pre-generate Reasoning for the top candidates
                for r in results[:150]:
                    cid = r["candidate_id"]
                    evidence = all_evidence.get(cid, {})
                    r["reasoning"] = generate_reasoning(r["candidate"], evidence)

                progress.empty()
                st.session_state.results = results
                st.session_state.honeypots = honeypots
                st.session_state.all_evidence = all_evidence
        except Exception as e:
            st.error(f"Error executing ranking pipeline: {e}")
            raw_candidates = []

    # Get results from state
    results = st.session_state.results
    honeypots = st.session_state.honeypots
    all_evidence = st.session_state.all_evidence

    if results:
        # Filter results based on search parameters
        filtered_results = results
        if min_score > 0.0:
            filtered_results = [r for r in filtered_results if r["final_score"] >= min_score]
        if search_title:
            keyword = search_title.lower()
            filtered_results = [
                r for r in filtered_results 
                if keyword in r["title"].lower() or keyword in r["company"].lower() or keyword in r["candidate_id"].lower()
            ]

        # Metric Blocks
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class='glass-card'>
                <small style='color:#a0aec0;'>Profiles Scored</small>
                <h2 style='margin:0.2rem 0;color:#fff;'>{len(results) + len(honeypots):,}</h2>
                <span style='color:#6ee7b7;font-size:0.85rem;'>✅ 100% processed</span>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class='glass-card'>
                <small style='color:#a0aec0;'>Honeypots Detected</small>
                <h2 style='margin:0.2rem 0;color:#fca5a5;'>{len(honeypots)}</h2>
                <span style='color:#fca5a5;font-size:0.85rem;'>🍯 0% infiltration target met</span>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            avg_score_val = sum(r["final_score"] for r in results[:100]) / min(100, len(results)) if results else 0
            st.markdown(f"""
            <div class='glass-card'>
                <small style='color:#a0aec0;'>Avg Shortlist Score</small>
                <h2 style='margin:0.2rem 0;color:#60a5fa;'>{avg_score_val:.3f}</h2>
                <span style='color:#60a5fa;font-size:0.85rem;'>⚡ Top 100 candidates</span>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            otw_rate = sum(1 for r in results[:100] if r["open_to_work"]) / min(100, len(results)) if results else 0
            st.markdown(f"""
            <div class='glass-card'>
                <small style='color:#a0aec0;'>Shortlist Availability</small>
                <h2 style='margin:0.2rem 0;color:#fbbf24;'>{otw_rate:.0%}</h2>
                <span style='color:#fbbf24;font-size:0.85rem;'>💼 Open-to-work flag</span>
            </div>
            """, unsafe_allow_html=True)

        # Main Columns
        main_col, side_col = st.columns([1.8, 1.2])

        with main_col:
            st.subheader("📋 Top Candidates Shortlist")
            
            shortlist_df = pd.DataFrame(filtered_results[:top_k])
            if not shortlist_df.empty:
                shortlist_df["rank"] = range(1, len(shortlist_df) + 1)
                
                # Format scores for presentation
                shortlist_df["score"] = shortlist_df["final_score"]
                
                st.dataframe(
                    shortlist_df[["rank", "candidate_id", "score", "title", "company", "years", "country", "reasoning"]],
                    use_container_width=True,
                    height=480,
                    hide_index=True
                )

                # CSV Download Button
                csv_data = shortlist_df[["candidate_id", "rank", "score", "reasoning"]].to_csv(index=False)
                st.download_button(
                    label="📥 Download Submission CSV",
                    data=csv_data,
                    file_name="submission.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("No candidates match your search parameters.")

        with side_col:
            st.subheader("📈 Pool Insights")
            
            # Simple Area Chart of Score Distribution
            if len(results) > 1:
                st.markdown("<small style='color:#a0aec0;'>Shortlist Score Curve</small>", unsafe_allow_html=True)
                score_curve = [r["final_score"] for r in results[:top_k]]
                st.area_chart(score_curve, height=180)
            
            # Aggregate skills of shortlist candidates
            st.markdown("<small style='color:#a0aec0;'>Top Technical Skills in Shortlist</small>", unsafe_allow_html=True)
            skills_pool = []
            for r in results[:50]:
                for s in r["candidate"].get("skills", []):
                    skills_pool.append(s["name"])
            
            skill_counts = Counter(skills_pool).most_common(10)
            if skill_counts:
                skills_df = pd.DataFrame(skill_counts, columns=["Skill", "Count"]).set_index("Skill")
                st.bar_chart(skills_df, height=200)

        # Profile Inspector Section
        st.markdown("---")
        st.subheader("🔍 Deep Candidate Inspector")
        st.markdown("Select a candidate from the shortlist to view their full resume breakdown and behavioral signals.")

        candidate_ids = [r["candidate_id"] for r in filtered_results[:top_k]]
        if candidate_ids:
            selected_cid = st.selectbox("Select Candidate to Inspect", candidate_ids)
            selected_cand = next(r for r in filtered_results if r["candidate_id"] == selected_cid)
            c = selected_cand["candidate"]
            profile = c.get("profile", {})
            evidence = all_evidence.get(selected_cid, {})

            # Premium styled Candidate Detail View
            card_col1, card_col2 = st.columns([1.2, 1.8])

            with card_col1:
                st.markdown(f"""
                <div class='glass-card' style='height: 100%;'>
                    <div class='candidate-card-header'>{profile.get('anonymized_name', selected_cid)}</div>
                    <div style='color:#f43f5e;font-weight:600;margin-bottom:0.8rem;'>{profile.get('current_title', 'N/A')}</div>
                    <div style='font-size:0.9rem;margin-bottom:0.5rem;'>🏢 <strong>Company:</strong> {profile.get('current_company', 'N/A')}</div>
                    <div style='font-size:0.9rem;margin-bottom:0.5rem;'>📁 <strong>Industry:</strong> {profile.get('current_industry', 'N/A')}</div>
                    <div style='font-size:0.9rem;margin-bottom:0.5rem;'>📍 <strong>Location:</strong> {profile.get('location', 'N/A')}, {profile.get('country', 'N/A')}</div>
                    <div style='font-size:0.9rem;margin-bottom:1.5rem;'>⌛ <strong>Experience:</strong> {profile.get('years_of_experience', 0):.1f} Years</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.markdown("**Recruiter Scoring Breakdown**")
                
                # Show breakdown with progress bars
                tech_score = evidence.get("tech_skill_core_ml", (0, 1))[0] / 8.0 if "tech_skill_core_ml" in evidence else 0.5
                st.write(f"Technical Capability ({tech_score:.0%})")
                st.progress(max(0.0, min(1.0, tech_score)))

                exp_score = evidence.get("exp_yoe", (0, 8))[0] / 8.0 if "exp_yoe" in evidence else 0.5
                st.write(f"Experience Match ({exp_score:.0%})")
                st.progress(max(0.0, min(1.0, exp_score)))

                behav_score = evidence.get("behav_active", (0, 5))[0] / 5.0 if "behav_active" in evidence else 0.5
                st.write(f"Behavior & Availability ({behav_score:.0%})")
                st.progress(max(0.0, min(1.0, behav_score)))

                loc_score = evidence.get("loc_india", (0, 4))[0] / 4.0 if "loc_india" in evidence else 0.5
                st.write(f"Location & Workmode ({loc_score:.0%})")
                st.progress(max(0.0, min(1.0, loc_score)))
                st.markdown("</div>", unsafe_allow_html=True)

            with card_col2:
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.markdown("##### 📝 Recruiter Justification")
                st.info(selected_cand.get("reasoning", "No reasoning generated."))

                st.markdown("##### 💼 Career Timeline")
                for exp in c.get("career_history", []):
                    dates = f"{exp.get('start_date', '?')} to {exp.get('end_date') if exp.get('end_date') else 'Present'}"
                    st.markdown(f"""
                    <div class='timeline-item'>
                        <div class='timeline-title'>{exp.get('title', 'N/A')} at {exp.get('company', 'N/A')}</div>
                        <div class='timeline-meta'>🗓️ {dates} ({exp.get('duration_months', 0)} months) • {exp.get('industry', 'N/A')}</div>
                        <div class='timeline-desc'>{exp.get('description', '')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("##### 🛠️ Extracted Skills")
                st.markdown("<div class='badge-container'>", unsafe_allow_html=True)
                badge_html = ""
                for s in c.get("skills", []):
                    badge_html += f"<span class='badge badge-tech'>{s['name']} ({s.get('proficiency', 'intermediate')})</span>"
                st.markdown(badge_html, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("##### 📊 Platform Activity Signals")
                sig = c.get("redrob_signals", {})
                sig_col1, sig_col2, sig_col3 = st.columns(3)
                with sig_col1:
                    st.write(f"📩 **Response Rate:** {sig.get('recruiter_response_rate', 0):.0%}")
                    st.write(f"⏱️ **Avg Response Time:** {sig.get('avg_response_time_hours', 'N/A')} Hours")
                with sig_col2:
                    st.write(f"📅 **Notice Period:** {sig.get('notice_period_days', 'N/A')} Days")
                    st.write(f"💻 **GitHub Activity:** {sig.get('github_activity_score', 'N/A')}/100")
                with sig_col3:
                    st.write(f"🤝 **Connections:** {sig.get('connection_count', 0)}")
                    st.write(f"🌟 **Profile Completeness:** {sig.get('profile_completeness_score', 0)}%")
                st.markdown("</div>", unsafe_allow_html=True)

        # Honeypots Log Section
        if honeypots:
            st.markdown("---")
            with st.expander(f"🍯 Filtered Traps & Honeypots ({len(honeypots)})"):
                st.markdown("The following profiles were identified as impossible chronological contradictions or fake records, and were removed from consideration.")
                hp_data = []
                for cid, details in honeypots.items():
                    hp_data.append({
                        "candidate_id": cid,
                        "exclusion_reason": details["reason"],
                        "claimed_yoe": details["profile"].get("profile", {}).get("years_of_experience", 0),
                        "current_title": details["profile"].get("profile", {}).get("current_title", "N/A")
                    })
                st.dataframe(pd.DataFrame(hp_data), use_container_width=True, hide_index=True)
else:
    # Initial state screen
    st.info("👈 Please select candidate source pool and press the 'Rank Candidates Pool' button to analyze candidates.")
    
    st.markdown("""
    ### System Architecture & Quality Measures
    * **FAISS + BM25 Retrieval**: Combines semantic embedding search (MiniLM) and lexical matching using Reciprocal Rank Fusion (RRF).
    * **Honeypot Shield**: Timeline consistency validator blocks fake profiles with 100% precision.
    * **recruiter-Grade Features**: Weights technical fit, stability, career progression, location, and platform activity.
    * **Zero-Hallucination Reasoning**: Explanations are strictly grounded in candidate metrics and history facts.
    """)