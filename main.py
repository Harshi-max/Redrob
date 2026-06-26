"""
Enhanced main.py - Complete ranking pipeline orchestration.

This is the entry point for the Intelligent Candidate Discovery & Ranking System.

Usage:
    python main.py --candidates candidates.jsonl --jd job_description.txt --out results.csv
"""

import argparse
import sys
import os
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Intelligent Candidate Discovery & Ranking System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with existing artifacts (offline)
  python main.py --candidates candidates.jsonl --artifacts artifacts/ --out results.csv
  
  # Pre-compute with API access
  python main.py --candidates candidates.jsonl --jd job_description.txt --precompute
        """
    )
    
    parser.add_argument(
        '--candidates',
        required=True,
        type=str,
        help='Path to candidates.jsonl file'
    )
    
    parser.add_argument(
        '--jd',
        type=str,
        help='Path to job description text file'
    )
    
    parser.add_argument(
        '--artifacts',
        type=str,
        default='./artifacts',
        help='Path to pre-computed artifacts directory (default: ./artifacts)'
    )
    
    parser.add_argument(
        '--out',
        type=str,
        default='./submission.csv',
        help='Output CSV path (default: ./submission.csv)'
    )
    
    parser.add_argument(
        '--precompute',
        action='store_true',
        help='Run pre-computation phase (requires JD and API access)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=100,
        help='Number of top candidates to return (default: 100)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("="*70)
    logger.info("Intelligent Candidate Discovery & Ranking System")
    logger.info("="*70)
    
    # Import after setting up logging
    from src.loader import load_candidates
    from src.pipeline import run_ranking
    
    # Validate inputs
    if not os.path.exists(args.candidates):
        logger.error(f"Candidates file not found: {args.candidates}")
        sys.exit(1)
    
    # Phase 1: Pre-computation (if needed)
    if args.precompute:
        if not args.jd or not os.path.exists(args.jd):
            logger.error("Pre-computation requires --jd argument with job description file")
            sys.exit(1)
        
        logger.info("Phase 1: Pre-computation")
        logger.info(f"- Candidates: {args.candidates}")
        logger.info(f"- Job Description: {args.jd}")
        logger.info(f"- Output Artifacts: {args.artifacts}")
        
        from src.pipeline import run_precompute
        
        start = time.time()
        run_precompute(
            candidates_path=args.candidates,
            jd_path=args.jd,
            output_dir=args.artifacts,
        )
        elapsed = time.time() - start
        logger.info(f"Pre-computation completed in {elapsed:.2f}s")
    
    # Phase 2: Ranking (main)
    logger.info("Phase 2: Ranking")
    logger.info(f"- Candidates: {args.candidates}")
    logger.info(f"- Artifacts: {args.artifacts}")
    logger.info(f"- Output: {args.out}")
    logger.info(f"- Top K: {args.top_k}")
    
    # Verify artifacts exist
    if not os.path.exists(args.artifacts):
        logger.error(f"Artifacts directory not found: {args.artifacts}")
        logger.error("Run with --precompute first, or ensure artifacts exist")
        sys.exit(1)
    
    start = time.time()
    
    try:
        run_ranking(
            artifacts_dir=args.artifacts,
            candidates_path=args.candidates,
            output_csv=args.out,
            top_k=args.top_k,
        )
        
        elapsed = time.time() - start
        logger.info(f"Ranking completed in {elapsed:.2f}s")
        logger.info(f"Results saved to: {args.out}")
        
        # Show summary
        import pandas as pd
        results = pd.read_csv(args.out)
        logger.info(f"Top 5 candidates:")
        for idx, row in results.head(5).iterrows():
            logger.info(f"  {row['rank']:3d}. {row['candidate_id']:20s} (score: {row['score']:.3f})")
        
        logger.info("="*70)
        logger.info("SUCCESS: Ranking pipeline completed successfully")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"Error during ranking: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
