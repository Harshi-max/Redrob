"""
Utils Module - Common utilities and helper functions.
"""

import json
import logging
from typing import Any, Dict, List
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_json(path: str) -> Any:
    """Load JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, path: str) -> None:
    """Save data to JSON file."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], path: str) -> None:
    """Save list of dicts to JSONL file."""
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_text(path: str) -> str:
    """Load text file."""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def save_text(text: str, path: str) -> None:
    """Save text to file."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)


def ensure_directory(path: str) -> Path:
    """Ensure directory exists."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def normalize_name(name: str) -> str:
    """Normalize text (lowercase, strip whitespace)."""
    return name.lower().strip()


def calculate_percentile(value: float, values: List[float]) -> float:
    """Calculate percentile rank of a value in a list."""
    sorted_vals = sorted(values)
    rank = len([v for v in sorted_vals if v <= value])
    return rank / len(values) if values else 0


def batch_process(items: List[Any], batch_size: int = 32):
    """Generator for batch processing."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]
