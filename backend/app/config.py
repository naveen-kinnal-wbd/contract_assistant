"""
Configuration loader for the contract assistant backend
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

# Config file path
CONFIG_FILE = Path(__file__).parent.parent / "config.json"


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """Load configuration from config.json (cached after first load)"""
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)


def get_upload_config() -> dict[str, Any]:
    """Get upload configuration"""
    config = load_config()
    return config.get("upload", {})


def get_llm_config() -> dict[str, Any]:
    """Get LLM configuration"""
    config = load_config()
    return config.get("llm", {})


def get_processing_config() -> dict[str, Any]:
    """Get processing configuration for image conversion"""
    config = load_config()
    return config.get("processing", {})

