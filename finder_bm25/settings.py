from __future__ import annotations

import os
from pathlib import Path

from .config import AppConfig

# Single place to configure the full pipeline.
CONFIG = AppConfig(
    root=Path(".").resolve(),
    train_subdir="dataset/train",
    tenk_subdir="dataset/10k",
    output_subdir="outputs",
    bm25_k1=1.2,
    bm25_b=0.75,
    top_k=10,    
    retrieval_method="bm25",  # bm25 | hyde | faiss | multipath
    embedding_model="BAAI/bge-base-en-v1.5",
    hyde_llm_model="Qwen/Qwen2.5-7B-Instruct",
    hyde_device="cuda",
    hyde_num_hypothetical_docs=3,
    chunk_expansion_similarity_threshold=0.85,
    chunk_expansion_max_neighbors=5,
    rrf_k=60,
    multipath_candidates_per_retriever=30,
    metadata_header=True,  # False = ablation tanpa header metadata (setara korpus v2)
    company_filter=False,  # True = ablation "company routing" (tidak sebanding dengan baseline FINDER)
    eval_mode="ragas_llm",  # heuristic | ragas_llm | ragas_nonllm
    fuzzy_threshold=0.85,
    sample_percent=10.0,
    sample_seed=42,
    no_progress=False,
    openrouter_endpoint="https://openrouter.ai/api/v1",
    openrouter_model="deepseek/deepseek-v4.1-flash",
    openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
    experiment_name="",
)
