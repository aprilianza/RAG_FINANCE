from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# Naikkan jika format korpus berubah agar cache korpus & embedding lama tidak terpakai.
# v2: tabel 10-K disimpan sebagai teks (sebelumnya dibuang).
# v3: tiap chunk menyimpan header metadata (perusahaan, ticker, tahun fiskal, Item 10-K).
CORPUS_VERSION = "v3"


def load_env(env_path: Path | None = None) -> None:
    """Load key-value pairs from .env into os.environ if not already set."""
    if env_path is None:
        env_path = Path(".env")
    if not env_path.is_file():
        parent_env = Path(__file__).resolve().parent.parent / ".env"
        if parent_env.is_file():
            env_path = parent_env
        else:
            return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key and key not in os.environ:
                        os.environ[key] = val
    except Exception:
        pass


load_env()


@dataclass
class AppConfig:
    # Paths
    root: Path = Path(".")
    train_subdir: str = "dataset/train"
    tenk_subdir: str = "dataset/10k"
    output_subdir: str = "outputs"

    # Retrieval config
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    top_k: int = 10
    retrieval_method: str = "bm25"  # bm25 | hyde | faiss | multipath
    embedding_model: str = "all-MiniLM-L6-v2"  # For HyDE, FAISS, and multipath
    hyde_llm_model: str = "Qwen/Qwen2.5-1.5B"  # For HyDE
    hyde_device: str = "cpu"  # cpu | cuda
    hyde_num_hypothetical_docs: int = 3
    
    # Chunk bundling config (for multipath retrieval)
    chunk_expansion_similarity_threshold: float = 0.85
    chunk_expansion_max_neighbors: int = 5
    # Reciprocal Rank Fusion (multipath)
    rrf_k: int = 60
    multipath_candidates_per_retriever: int = 30

    # Index BM25/embedding dengan header metadata dokumen di depan tiap chunk
    # (enrichment ala FinSage FFP). Pencarian tetap di seluruh korpus.
    metadata_header: bool = True

    # Versi format korpus; run lama yang dibaca ulang (rescore) memakai versinya sendiri.
    corpus_version: str = CORPUS_VERSION

    # Batasi retrieval ke perusahaan yang disebut di query (ticker / nama).
    company_filter: bool = False

    # Eval config
    eval_mode: str = "heuristic"  # heuristic | ragas_llm | ragas_nonllm
    fuzzy_threshold: float = 0.85

    # Stratified sampling config
    sample_percent: float = 100.0
    sample_seed: int = 42

    # Progress display
    no_progress: bool = False

    # OpenRouter config
    openrouter_endpoint: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "nvidia/nemotron-3-nano-30b-a3b:free"
    openrouter_api_key: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY", "")
    )
    # Optional experiment name; outputs will be written under output_dir/experiment_name/
    experiment_name: str = ""

    target_categories: List[str] = field(
        default_factory=lambda: [
            "Accounting",
            "Company Overview",
            "Financials",
            "Footnotes",
            "Governance",
            "Legal",
            "Risk",
            "Shareholder Return",
        ]
    )

    @property
    def train_dir(self) -> Path:
        return self.root / self.train_subdir

    @property
    def tenk_dir(self) -> Path:
        return self.root / self.tenk_subdir

    @property
    def output_dir(self) -> Path:
        return self.root / self.output_subdir

    @property
    def corpus_cache_path(self) -> Path:
        suffix = f"_{self.corpus_version}" if self.corpus_version else ""
        return self.output_dir / f"corpus_paragraphs{suffix}.jsonl"

    @property
    def company_names_cache_path(self) -> Path:
        return self.output_dir / "company_names.json"

    @property
    def faiss_embeddings_cache_path(self) -> Path:
        """Cache path untuk FAISS embeddings (numpy format)."""
        # Encode model name untuk avoid path issues (ganti '/' dengan '_')
        model_safe = self.embedding_model.replace("/", "_").replace("-", "_")
        return self.output_dir / f"faiss_embeddings_{model_safe}_{self.corpus_version}{self._header_suffix}.npy"

    @property
    def hyde_embeddings_cache_path(self) -> Path:
        """Cache path untuk HyDE embeddings (numpy format)."""
        model_safe = self.embedding_model.replace("/", "_").replace("-", "_")
        return self.output_dir / f"hyde_embeddings_{model_safe}_{self.corpus_version}{self._header_suffix}.npy"

    @property
    def _header_suffix(self) -> str:
        # Embedding dengan dan tanpa header metadata disimpan terpisah.
        return "_hdr" if self.metadata_header else ""

    @property
    def retrieval_details_path(self) -> Path:
        return self.output_dir / "retrieval_details.jsonl"

    @property
    def markdown_out_path(self) -> Path:
        return self.output_dir / "bm25_context_recall.md"

    @property
    def csv_out_path(self) -> Path:
        return self.output_dir / "bm25_context_recall.csv"


def validate_config(config: AppConfig) -> None:
    if config.top_k <= 0:
        raise ValueError("top_k must be > 0")
    if config.sample_percent <= 0 or config.sample_percent > 100:
        raise ValueError("sample_percent must be in (0, 100]")
    if config.eval_mode not in {"heuristic", "ragas_llm", "ragas_nonllm"}:
        raise ValueError("eval_mode must be one of: heuristic, ragas_llm, ragas_nonllm")
    if config.retrieval_method not in {"bm25", "hyde", "faiss", "multipath"}:
        raise ValueError("retrieval_method must be one of: bm25, hyde, faiss, multipath")
    if not (0.0 <= config.chunk_expansion_similarity_threshold <= 1.0):
        raise ValueError("chunk_expansion_similarity_threshold must be between 0.0 and 1.0")
    if config.chunk_expansion_max_neighbors <= 0:
        raise ValueError("chunk_expansion_max_neighbors must be > 0")
    if config.rrf_k <= 0:
        raise ValueError("rrf_k must be > 0")
    if config.multipath_candidates_per_retriever <= 0:
        raise ValueError("multipath_candidates_per_retriever must be > 0")
