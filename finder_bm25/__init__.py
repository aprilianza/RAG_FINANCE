"""FINDER BM25 baseline package."""

from .config import AppConfig
from .pipeline import run_pipeline

__all__ = ["AppConfig", "run_pipeline"]
