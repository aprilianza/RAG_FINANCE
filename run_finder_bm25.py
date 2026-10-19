#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from finder_bm25 import AppConfig, run_pipeline
from finder_bm25.settings import CONFIG


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Modular BM25, FAISS dense, and HyDE retrieval pipeline for FINDER"
    )
    parser.add_argument("--root", type=str, default=None, help="Optional override for workspace root")
    parser.add_argument("--eval-mode", type=str, default=None, choices=["heuristic", "ragas_llm", "ragas_nonllm"])
    parser.add_argument("--sample-percent", type=float, default=None, help="Optional override for stratified sample percentage")
    parser.add_argument("--sample-seed", type=int, default=None, help="Optional override for random seed")
    parser.add_argument("--top-k", type=int, default=None, help="Optional override for Top-K retrieval")
    parser.add_argument("--k1", type=float, default=None, help="Optional override for BM25 k1")
    parser.add_argument("--b", type=float, default=None, help="Optional override for BM25 b")
    parser.add_argument("--fuzzy-threshold", type=float, default=None, help="Optional override for heuristic fuzzy threshold")
    parser.add_argument("--openrouter-endpoint", type=str, default=None)
    parser.add_argument("--openrouter-model", type=str, default=None)
    parser.add_argument("--openrouter-api-key", type=str, default=None)
    parser.add_argument("--no-progress", action="store_true", help="Optional override to disable tqdm")
    parser.add_argument("--experiment-name", type=str, default=None, help="Optional experiment folder name under outputs to store run files")
    parser.add_argument("--retrieval-method", type=str, default=None, choices=["bm25", "hyde", "faiss", "multipath"], help="Retrieval method: bm25, hyde, faiss, or multipath")
    parser.add_argument("--embedding-model", type=str, default=None, help="Sentence transformer model for HyDE and FAISS embeddings")
    parser.add_argument("--hyde-llm-model", type=str, default=None, help="Hugging Face model for HyDE LLM")
    parser.add_argument("--hyde-device", type=str, default=None, choices=["cpu", "cuda"], help="Device for HyDE and FAISS dense retrieval (cpu or cuda)")
    parser.add_argument("--hyde-num-hypothetical-docs", type=int, default=None, help="Number of hypothetical docs for HyDE")
    parser.add_argument("--chunk-expansion-similarity-threshold", type=float, default=None, help="Similarity threshold for chunk bundling in multipath retrieval (0.0-1.0)")
    parser.add_argument("--chunk-expansion-max-neighbors", type=int, default=None, help="Max neighbors to check for chunk bundling in multipath retrieval")
    parser.add_argument("--rrf-k", type=int, default=None, help="Reciprocal Rank Fusion constant for multipath retrieval")
    parser.add_argument("--multipath-candidates", type=int, default=None, help="Candidates per retriever before RRF fusion in multipath retrieval")
    parser.add_argument(
        "--metadata-header",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Index chunks with a company/fiscal-year/10-K-section header (default on). --no-metadata-header for the ablation",
    )
    parser.add_argument(
        "--company-filter",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Ablation: restrict retrieval to companies mentioned in the query (not comparable with FINDER baselines). Default off",
    )
    return parser


def build_config(args: argparse.Namespace) -> AppConfig:
    # Start from single source of truth in finder_bm25/settings.py
    config = AppConfig(**CONFIG.__dict__)

    # Optional CLI overrides.
    if args.root is not None:
        config.root = Path(args.root).resolve()
    if args.eval_mode is not None:
        config.eval_mode = args.eval_mode
    if args.sample_percent is not None:
        config.sample_percent = args.sample_percent
    if args.sample_seed is not None:
        config.sample_seed = args.sample_seed
    if args.top_k is not None:
        config.top_k = args.top_k
    if args.k1 is not None:
        config.bm25_k1 = args.k1
    if args.b is not None:
        config.bm25_b = args.b
    if args.fuzzy_threshold is not None:
        config.fuzzy_threshold = args.fuzzy_threshold
    if args.openrouter_endpoint is not None:
        config.openrouter_endpoint = args.openrouter_endpoint
    if args.openrouter_model is not None:
        config.openrouter_model = args.openrouter_model
    if args.openrouter_api_key is not None:
        config.openrouter_api_key = args.openrouter_api_key
    if args.no_progress:
        config.no_progress = True
    if args.experiment_name is not None:
        config.experiment_name = args.experiment_name
    if args.retrieval_method is not None:
        config.retrieval_method = args.retrieval_method
    if args.embedding_model is not None:
        config.embedding_model = args.embedding_model
    if args.hyde_llm_model is not None:
        config.hyde_llm_model = args.hyde_llm_model
    if args.hyde_device is not None:
        config.hyde_device = args.hyde_device
    if args.hyde_num_hypothetical_docs is not None:
        config.hyde_num_hypothetical_docs = args.hyde_num_hypothetical_docs
    if args.chunk_expansion_similarity_threshold is not None:
        config.chunk_expansion_similarity_threshold = args.chunk_expansion_similarity_threshold
    if args.chunk_expansion_max_neighbors is not None:
        config.chunk_expansion_max_neighbors = args.chunk_expansion_max_neighbors

    if args.rrf_k is not None:
        config.rrf_k = args.rrf_k
    if args.multipath_candidates is not None:
        config.multipath_candidates_per_retriever = args.multipath_candidates
    if args.metadata_header is not None:
        config.metadata_header = args.metadata_header
    if args.company_filter is not None:
        config.company_filter = args.company_filter

    return config


def main() -> None:
    args = build_parser().parse_args()
    run_pipeline(build_config(args))


if __name__ == "__main__":
    main()
