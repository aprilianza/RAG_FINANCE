from __future__ import annotations

import re
import numpy as np
from typing import Any, Sequence

# ── Optional dependencies — masing-masing blok terpisah ──────────────────────

try:
    from concurrent.futures import ThreadPoolExecutor, as_completed  # type: ignore[import-not-found]
except Exception:
    ThreadPoolExecutor = None
    as_completed = None

try:
    from rank_bm25 import BM25Okapi  # type: ignore[import-not-found]
except Exception:
    BM25Okapi = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
except Exception:
    SentenceTransformer = None

try:
    import faiss  # type: ignore[import-not-found]
except Exception:
    faiss = None

# FIX: Pisah transformers ke blok sendiri agar tidak ikut gagal
try:
    import torch
    from transformers import (  # type: ignore[import-not-found]
        AutoTokenizer,
        AutoModelForCausalLM,
        pipeline as hf_pipeline_factory,
    )
    _TRANSFORMERS_AVAILABLE = True
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    hf_pipeline_factory = None
    _TRANSFORMERS_AVAILABLE = False

# FIX: Coba import langchain_huggingface dulu, fallback ke langchain.llms lama
try:
    from langchain_huggingface import HuggingFacePipeline  # type: ignore[import-not-found]
    _LANGCHAIN_HF_AVAILABLE = True
except Exception:
    try:
        from langchain.llms import HuggingFacePipeline  # type: ignore[import-not-found]
        _LANGCHAIN_HF_AVAILABLE = True
    except Exception:
        HuggingFacePipeline = None
        _LANGCHAIN_HF_AVAILABLE = False

from .preprocess import Paragraph
from .utils import tokenize, tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Konstanta HyDE
# ─────────────────────────────────────────────────────────────────────────────

# Panjang ideal hypothetical doc untuk HyDE embedding
# Referensi paper FinSage: hypothetical doc harus singkat & padat
_HYDE_MAX_CHARS      = 600   # batas karakter output
_HYDE_MAX_NEW_TOKENS = 150    # ~80 token ≈ 2-3 kalimat — FIX konflik max_length
_HYDE_MIN_NEW_TOKENS = 50    # hindari output kosong


# ─────────────────────────────────────────────────────────────────────────────
# Utilitas internal
# ─────────────────────────────────────────────────────────────────────────────

def _truncate_to_sentence(text: str, max_chars: int = _HYDE_MAX_CHARS) -> str:
    """
    Potong teks di kalimat lengkap terakhir sebelum batas max_chars.
    Jika tidak ada titik/tanda akhir kalimat, potong paksa di max_chars.
    """
    if len(text) <= max_chars:
        return text
    snippet = text[:max_chars]
    last_end = max(snippet.rfind("."), snippet.rfind("!"), snippet.rfind("?"))
    if last_end > max_chars // 2:
        return text[:last_end + 1].strip()
    return snippet.strip()


def _clean_generated_text(text: str) -> str:
    """
    Bersihkan artefak umum dari output LLM:
    - Hapus newline berlebih
    - Hapus karakter non-printable
    - Normalisasi whitespace
    """
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[^\x20-\x7E\n]", "", text)
    text = " ".join(text.split())
    return text.strip()


def _encode_normalized_embeddings(
    embedding_model: Any,
    texts: Sequence[str],
    show_progress_bar: bool,
) -> np.ndarray:
    embeddings = embedding_model.encode(
        list(texts),
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        batch_size=32,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def _load_or_compute_embeddings(
    embedding_model: Any,
    texts: Sequence[str],
    cache_path: Any,
    show_progress_bar: bool = True,
) -> np.ndarray:
    """
    Load embeddings dari cache jika ada, atau compute dan save ke cache.
    
    Args:
        embedding_model:  Sentence Transformer model.
        texts:            Teks-teks untuk di-embed.
        cache_path:       Path untuk cache embeddings (numpy format).
        show_progress_bar: Tampilkan progress bar saat encode.
    
    Returns:
        Embeddings dalam bentuk numpy array (float32, normalized).
    """
    # Check cache
    if cache_path.exists():
        try:
            print(f"[INFO] Loading cached embeddings from: {cache_path}")
            embeddings = np.load(cache_path)
            print(f"[INFO] Loaded {len(embeddings)} cached embeddings")
            return embeddings
        except Exception as e:
            print(f"[WARN] Failed to load embeddings cache: {e}, recomputing...")

    # Compute embeddings
    print(f"[INFO] Computing embeddings for {len(texts)} texts...")
    embeddings = _encode_normalized_embeddings(
        embedding_model,
        texts,
        show_progress_bar=show_progress_bar,
    )

    # Save to cache
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Saving embeddings cache to: {cache_path}")
        np.save(cache_path, embeddings)
        print(f"[INFO] Saved {len(embeddings)} embeddings to cache")
    except Exception as e:
        print(f"[WARN] Failed to save embeddings cache: {e}")

    return embeddings


# ─────────────────────────────────────────────────────────────────────────────
# BM25
# ─────────────────────────────────────────────────────────────────────────────

def build_bm25(
    paragraphs: Sequence[Paragraph],
    k1: float = 1.2,
    b: float = 0.75,
) -> tuple[Any, list[list[str]]]:
    """Build BM25 index dari corpus paragraf."""
    if BM25Okapi is None:
        raise RuntimeError(
            "Missing dependency 'rank_bm25'. "
            "Install with: pip install rank-bm25"
        )

    tokenized_corpus = [
        tokenize(p.text)
        for p in tqdm(paragraphs, desc="Tokenize corpus for BM25")
    ]
    bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)
    return bm25, tokenized_corpus


def retrieve_topk(
    bm25: Any,
    paragraphs: Sequence[Paragraph],
    query: str,
    k: int = 10,
) -> list[tuple[Paragraph, float]]:
    """Retrieve top-k paragraf menggunakan BM25."""
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)
    idx_scores = list(enumerate(scores))
    idx_scores.sort(key=lambda x: (-x[1], x[0]))
    top = idx_scores[:k]
    return [(paragraphs[i], float(s)) for i, s in top]


# ─────────────────────────────────────────────────────────────────────────────
# HyDE — Hypothetical Document Embeddings
# ─────────────────────────────────────────────────────────────────────────────

def build_hyde_llm(
    llm_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    device: str = "cpu",
) -> Any:
    """
    Load HuggingFace LLM pipeline sekali saja untuk reuse di semua queries.
    Pipeline di-cache agar tidak di-load ulang tiap query.

    FIX utama di fungsi ini:
      - model_max_length=512 di tokenizer → hilangkan warning konflik max_length
      - Hanya pakai max_new_tokens, TIDAK pakai max_length di pipeline
      - pad_token_id eksplisit → hilangkan warning padding
      - repetition_penalty → kurangi output yang berulang & panjang
      - model.eval() → hemat memori, nonaktifkan dropout

    Args:
        llm_model_name:  HuggingFace model name.
        device:          'cpu' atau 'cuda'.

    Returns:
        Transformers text-generation pipeline (reusable).
    """
    if not _TRANSFORMERS_AVAILABLE:
        raise RuntimeError(
            "Missing dependency 'transformers'. "
            "Install with: pip install transformers accelerate"
        )

    print(f"[INFO] Loading LLM for HyDE: {llm_model_name}")

    # FIX 1: Set model_max_length eksplisit
    # Ini menghilangkan: "Both max_new_tokens and max_length seem to have been set"
    # karena beberapa tokenizer punya model_max_length=20 dari config lama
    tokenizer = AutoTokenizer.from_pretrained(
        llm_model_name,
        model_max_length=512,
        trust_remote_code=True,
    )

    # FIX 2: Pastikan pad_token ada
    # Qwen dan beberapa model tidak mendefinisikan pad_token secara default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Pilih dtype & device_map berdasarkan ketersediaan GPU
    if device == "cuda" and torch is not None and torch.cuda.is_available():
        torch_dtype = torch.float16
        device_map  = "cuda"
        print("[INFO] Running LLM on CUDA (float16)")
    else:
        torch_dtype = torch.float32 if torch is not None else "auto"
        device_map  = "cpu"
        print("[INFO] Running LLM on CPU (float32)")

    model = AutoModelForCausalLM.from_pretrained(
        llm_model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # FIX 3: eval mode — nonaktifkan dropout, hemat memori inferensi
    model.eval()

    # FIX 4: Hanya set max_new_tokens, JANGAN set max_length
    # max_length = max_new_tokens + panjang_prompt → sering bentrok
    # max_new_tokens = hanya jumlah token OUTPUT baru → lebih aman
    gen_pipeline = hf_pipeline_factory(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=_HYDE_MAX_NEW_TOKENS,    # ~80 token = 2-3 kalimat pendek
        min_new_tokens=_HYDE_MIN_NEW_TOKENS,    # hindari output kosong
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.15,               # FIX 5: kurangi repetisi & panjang
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id,   # FIX 6: hilangkan warning padding
        # max_length TIDAK diset di sini — penyebab konflik sebelumnya
    )

    print(f"[INFO] LLM pipeline ready (max_new_tokens={_HYDE_MAX_NEW_TOKENS})")
    return gen_pipeline


def build_hyde_retriever(
    paragraphs: Sequence[Paragraph],
    model_name: str = "all-MiniLM-L6-v2",
    device: str = "cpu",
    cache_path: Any = None,
) -> tuple[Any, Any]:
    """
    Build HyDE retriever dengan local embedding model.

    Rekomendasi model (sesuai paper FinSage):
      - "BAAI/bge-m3"         : terbaik, sesuai paper (GPU ≥ 8GB)
      - "BAAI/bge-small-en-v1.5" : ringan, cukup baik (CPU)
      - "all-MiniLM-L6-v2"    : default fallback, paling ringan

    Args:
        paragraphs:  Corpus paragraf.
        model_name:  Sentence Transformer model untuk embedding.
        device:      'cpu' atau 'cuda'.
        cache_path:  Optional path untuk cache embeddings (numpy format).

    Returns:
        (embedding_model, paragraph_embeddings)
    """
    if SentenceTransformer is None:
        raise RuntimeError(
            "Missing dependency 'sentence-transformers'. "
            "Install with: pip install sentence-transformers"
        )

    print(f"[INFO] Loading embedding model: {model_name}")
    embedding_model = SentenceTransformer(model_name, device=device)

    paragraph_texts = [p.text for p in paragraphs]
    
    if cache_path is not None:
        paragraph_embeddings = _load_or_compute_embeddings(
            embedding_model,
            paragraph_texts,
            cache_path,
            show_progress_bar=True,
        )
    else:
        print(f"[INFO] Embedding {len(paragraphs)} corpus paragraphs (no cache)...")
        paragraph_embeddings = _encode_normalized_embeddings(
            embedding_model,
            paragraph_texts,
            show_progress_bar=True,
        )

    return embedding_model, paragraph_embeddings


def build_faiss_retriever(
    paragraphs: Sequence[Paragraph],
    model_name: str = "all-MiniLM-L6-v2",
    device: str = "cpu",
    cache_path: Any = None,
) -> tuple[Any, Any, np.ndarray]:
    """
    Build dense retriever berbasis FAISS dengan sentence-transformer embeddings.

    Embedding dinormalisasi terlebih dahulu agar IndexFlatIP dapat dipakai sebagai
    cosine-similarity search. Embeddings dapat di-cache untuk menghindari re-computation.
    
    Args:
        paragraphs:  Corpus paragraf.
        model_name:  Sentence Transformer model untuk embedding.
        device:      'cpu' atau 'cuda'.
        cache_path:  Optional path untuk cache embeddings (numpy format).
    
    Returns:
        (embedding_model, faiss_index, paragraph_embeddings)
    """
    if SentenceTransformer is None:
        raise RuntimeError(
            "Missing dependency 'sentence-transformers'. "
            "Install with: pip install sentence-transformers"
        )
    if faiss is None:
        raise RuntimeError(
            "Missing dependency 'faiss'. Install FAISS support before using retrieval_method='faiss'. "
            "On many setups you can use: pip install faiss-cpu"
        )

    if not paragraphs:
        raise ValueError("Cannot build FAISS retriever from an empty paragraph corpus")

    print(f"[INFO] Loading embedding model for FAISS: {model_name}")
    embedding_model = SentenceTransformer(model_name, device=device)

    paragraph_texts = [p.text for p in paragraphs]
    
    if cache_path is not None:
        paragraph_embeddings = _load_or_compute_embeddings(
            embedding_model,
            paragraph_texts,
            cache_path,
            show_progress_bar=True,
        )
    else:
        print(f"[INFO] Embedding {len(paragraphs)} corpus paragraphs for FAISS (no cache)...")
        paragraph_embeddings = _encode_normalized_embeddings(
            embedding_model,
            paragraph_texts,
            show_progress_bar=True,
        )

    print("[INFO] Building FAISS IndexFlatIP index...")
    index = faiss.IndexFlatIP(paragraph_embeddings.shape[1])
    index.add(paragraph_embeddings)

    return embedding_model, index, paragraph_embeddings


def retrieve_topk_faiss(
    embedding_model: Any,
    faiss_index: Any,
    paragraphs: Sequence[Paragraph],
    query: str,
    k: int = 10,
) -> list[tuple[Paragraph, float]]:
    """Retrieve top-k paragraf menggunakan FAISS dense retrieval."""
    if faiss is None:
        raise RuntimeError(
            "Missing dependency 'faiss'. Install FAISS support before using retrieval_method='faiss'. "
            "On many setups you can use: pip install faiss-cpu"
        )

    query_embedding = _encode_normalized_embeddings(
        embedding_model,
        [query],
        show_progress_bar=False,
    )

    top_k = min(k, len(paragraphs))
    scores, indices = faiss_index.search(query_embedding, top_k)

    results: list[tuple[Paragraph, float]] = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < 0:
            continue
        results.append((paragraphs[int(idx)], float(score)))

    return results


def generate_hypothetical_documents(
    query: str,
    gen_pipeline: Any,
    num_docs: int = 3,
) -> list[str]:
    """
    Generate hypothetical documents menggunakan pre-loaded LLM pipeline.

    Panjang ideal: 100-300 chars (~50-80 token).
    Referensi: FinSage paper Section 3.2 — HyDE Retriever.

    FIX di fungsi ini:
      - Prompt eksplisit meminta "2-3 kalimat" → kontrol panjang output
      - _clean_generated_text → hapus artefak LLM
      - _truncate_to_sentence → potong di kalimat lengkap jika terlalu panjang
      - Fallback ke query jika output kosong

    Args:
        query:        Query dari user.
        gen_pipeline: Text-generation pipeline (dari build_hyde_llm()).
        num_docs:     Jumlah hypothetical doc yang di-generate.

    Returns:
        List of generated hypothetical documents (sudah di-trim & dibersihkan).
    """
    # FIX: Prompt eksplisit "2-3 sentences" mendorong LLM generate singkat
    PROMPT_TEMPLATE = """You are an expert financial analyst and auditor analyzing SEC Form 10-K filings for S&P 500 companies.

    Your task is to write a highly realistic, hypothetical 4-5 sentence excerpt from an annual report that would contain the exact answer to the user's query.

    CRITICAL INSTRUCTIONS:
    1. Decode Jargon: The user's query may contain brief financial jargon, acronyms, or ambiguous tickers (e.g., "MS" for Morgan Stanley). Expand these internally before writing.
    2. Tone & Structure: Use the formal, precise, and legalistic tone typical of corporate 10-K disclosures.
    3. Quantitative Depth: If the query asks for a calculated metric, include hypothetical underlying components.

    Question: {query}
    Hypothetical 10-K Passage:
    """

    hypothetical_docs: list[str] = []

    for i in range(num_docs):
        prompt = PROMPT_TEMPLATE.format(query=query)

        try:
            output = gen_pipeline(prompt)[0]["generated_text"]
        except Exception as e:
            print(f"[WARN] LLM generation failed for doc {i+1}: {e}")
            # Fallback: gunakan query asli sebagai pseudo-hypothetical doc
            hypothetical_docs.append(query)
            continue

        # FIX: Bersihkan artefak LLM (newline berlebih, karakter aneh)
        generated = _clean_generated_text(output)

        # FIX: Potong di kalimat lengkap jika melewati batas
        generated = _truncate_to_sentence(generated, max_chars=_HYDE_MAX_CHARS)

        # Fallback jika output kosong setelah cleaning
        if not generated:
            print(f"[WARN] Empty output for doc {i+1}, using query as fallback")
            generated = query

        hypothetical_docs.append(generated)
        print(
            f"[INFO] Hyp doc {i + 1}/{num_docs}: "
            f"{len(generated)} chars | "
            f"~{len(generated.split())} words"
        )

    return hypothetical_docs


def retrieve_topk_hyde(
    embedding_model: Any,
    paragraph_embeddings: Any,
    paragraphs: Sequence[Paragraph],
    query: str,
    gen_pipeline: Any,
    k: int = 10,
    num_hypothetical_docs: int = 3,
) -> tuple[list[tuple[Paragraph, float]], list[str]]:
    """
    Retrieve top-k paragraf menggunakan metode HyDE.

    Alur (sesuai paper FinSage Section 3.2):
        1. Generate N hypothetical documents via LLM lokal.
        2. Embed setiap hypothetical doc (dengan normalize_embeddings=True).
        3. Hitung mean embedding, lalu re-normalize.
        4. Cosine similarity antara mean embedding vs paragraph_embeddings.
        5. Return top-k paragraf + hypothetical docs untuk debugging.

    FIX di fungsi ini:
      - normalize_embeddings=True pada hypothesis_embeddings (konsisten dengan corpus)
      - Re-normalize mean embedding setelah np.mean
      - Validasi: fallback ke query embedding jika semua hyp doc kosong

    Args:
        embedding_model:       Sentence Transformer (dari build_hyde_retriever).
        paragraph_embeddings:  Pre-computed embeddings corpus (dari build_hyde_retriever).
        paragraphs:            Corpus paragraf.
        query:                 Query dari user.
        gen_pipeline:          Pre-loaded LLM pipeline (dari build_hyde_llm).
        k:                     Jumlah hasil yang dikembalikan.
        num_hypothetical_docs: Jumlah hypothetical docs (paper FinSage: 3).

    Returns:
        Tuple of:
          - list[(Paragraph, similarity_score)] : top-k paragraf
          - list[str]                           : hypothetical docs (untuk debug)
    """
    from sklearn.metrics.pairwise import cosine_similarity  # lazy import

    # 1. Generate hypothetical documents
    print(f"[INFO] Generating {num_hypothetical_docs} hypothetical documents...")
    hypothetical_docs = generate_hypothetical_documents(
        query,
        gen_pipeline=gen_pipeline,
        num_docs=num_hypothetical_docs,
    )

    # Validasi: pastikan ada setidaknya 1 dokumen valid
    valid_docs = [d for d in hypothetical_docs if d and len(d.strip()) > 0]
    if not valid_docs:
        print("[WARN] All hypothetical docs empty, falling back to query embedding")
        valid_docs = [query]

    # 2. Embed hypothetical documents
    print("[INFO] Embedding hypothetical documents...")
    hypothesis_embeddings = embedding_model.encode(
        valid_docs,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,  # FIX: konsisten dengan corpus embeddings
    )

    # 3. Mean embedding (sesuai paper FinSage)
    mean_hypothesis_embedding = np.mean(hypothesis_embeddings, axis=0)

    # FIX: Re-normalize setelah mean agar cosine similarity tetap valid
    # (mean dari unit vectors bukan lagi unit vector)
    norm = np.linalg.norm(mean_hypothesis_embedding)
    if norm > 0:
        mean_hypothesis_embedding = mean_hypothesis_embedding / norm

    # 4. Cosine similarity vs seluruh corpus
    scores = cosine_similarity(
        [mean_hypothesis_embedding],
        paragraph_embeddings,
    )[0]

    # 5. Sort dan ambil top-k
    idx_scores = list(enumerate(scores))
    idx_scores.sort(key=lambda x: (-x[1], x[0]))
    top = idx_scores[:k]

    return [(paragraphs[i], float(s)) for i, s in top], hypothetical_docs


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Path Retrieval (MPR) — BM25 + FAISS + HyDE
# ─────────────────────────────────────────────────────────────────────────────

def _deduplicate_results(
    results: list[tuple[Paragraph, float, str]],
) -> list[tuple[Paragraph, float, list[str]]]:
    """
    Hapus duplikat berdasarkan paragraph_id.
    Jika ada duplikat, pertahankan score tertinggi dan track semua source retriever.

    Args:
        results: List of (Paragraph, score, source_method)

    Returns:
        List of (Paragraph, max_score, [all_sources])
    """
    seen: dict[str, tuple[Paragraph, float, list[str]]] = {}

    for para, score, source in results:
        para_id = para.paragraph_id
        if para_id not in seen:
            seen[para_id] = (para, score, [source])
        else:
            _, max_score, sources = seen[para_id]
            new_score = max(max_score, score)
            if source not in sources:
                sources.append(source)
            seen[para_id] = (para, new_score, sources)

    return [
        (para, score, sources)
        for para, score, sources in seen.values()
    ]


def _expand_chunks_with_bundling(
    deduplicated: list[tuple[Paragraph, float, list[str]]],
    paragraphs: Sequence[Paragraph],
    embedding_model: Any,
    similarity_threshold: float = 0.85,
    max_neighbors: int = 5,
) -> list[tuple[Paragraph, float, list[str], list[str]]]:
    """
    Chunk Expansion: Untuk setiap chunk yang dikembalikan, cek chunk tetangga 
    (sebelum dan sesudah) di dokumen asli dan gabungkan jika similaritasnya
    melebihi threshold.

    Returns:
        List of (bundled_Paragraph, original_score, sources, neighbor_ids)
    """
    if not deduplicated:
        return []

    # Build mapping: source_file -> list of (paragraph_id, index)
    doc_paragraphs: dict[str, list[tuple[str, int]]] = {}
    for idx, para in enumerate(paragraphs):
        if para.source_file not in doc_paragraphs:
            doc_paragraphs[para.source_file] = []
        doc_paragraphs[para.source_file].append((para.paragraph_id, idx))

    bundled_results: list[tuple[Paragraph, float, list[str], list[str]]] = []

    for para, score, sources in deduplicated:
        neighbor_texts: list[str] = [para.text]
        neighbor_ids: list[str] = [para.paragraph_id]

        # Cari posisi chunk ini di dokumen asli
        if para.source_file in doc_paragraphs:
            para_list = doc_paragraphs[para.source_file]
            try:
                current_pos = next(
                    i for i, (pid, _) in enumerate(para_list)
                    if pid == para.paragraph_id
                )

                # Query embedding untuk similarity check
                query_embedding = _encode_normalized_embeddings(
                    embedding_model,
                    [para.text],
                    show_progress_bar=False,
                )
                query_vec = query_embedding[0]

                # Cek neighbors sebelum dan sesudah
                neighbors_to_check = []
                for offset in range(1, max_neighbors + 1):
                    if current_pos - offset >= 0:
                        neighbors_to_check.append(
                            (current_pos - offset, "before")
                        )
                    if current_pos + offset < len(para_list):
                        neighbors_to_check.append(
                            (current_pos + offset, "after")
                        )

                for neighbor_pos, direction in neighbors_to_check:
                    neighbor_id, neighbor_idx = para_list[neighbor_pos]
                    neighbor_para = paragraphs[neighbor_idx]

                    # Compute similarity
                    neighbor_embedding = _encode_normalized_embeddings(
                        embedding_model,
                        [neighbor_para.text],
                        show_progress_bar=False,
                    )
                    neighbor_vec = neighbor_embedding[0]

                    # Cosine similarity (inner product of normalized vectors)
                    similarity = float(np.dot(query_vec, neighbor_vec))

                    if similarity > similarity_threshold:
                        if direction == "before":
                            neighbor_texts.insert(0, neighbor_para.text)
                            neighbor_ids.insert(0, neighbor_id)
                        else:
                            neighbor_texts.append(neighbor_para.text)
                            neighbor_ids.append(neighbor_id)

            except StopIteration:
                pass

        # Gabungkan teks dari neighbor menjadi satu bundled paragraph
        bundled_text = " ".join(neighbor_texts)
        bundled_para = Paragraph(
            paragraph_id=para.paragraph_id,
            source_file=para.source_file,
            text=bundled_text,
        )

        bundled_results.append((bundled_para, score, sources, neighbor_ids))

    return bundled_results


def retrieve_multipath(
    bm25: Any,
    embedding_model: Any,
    faiss_index: Any,
    paragraphs: Sequence[Paragraph],
    query: str,
    gen_pipeline: Any = None,
    k: int = 10,
    hyde_num_hypothetical_docs: int = 3,
    similarity_threshold: float = 0.85,
    max_neighbors: int = 5,
) -> list[tuple[Paragraph, float]]:
    """
    Multi-Path Retrieval (MPR): Menjalankan BM25, FAISS, dan HyDE secara paralel,
    menggabungkan hasil, menghapus duplikat, lalu melakukan chunk expansion/bundling.

    Args:
        bm25:                     BM25 index.
        embedding_model:          Sentence Transformer model.
        faiss_index:              FAISS index.
        paragraphs:               Corpus paragraf.
        query:                    Query string.
        gen_pipeline:             HyDE LLM pipeline (optional).
        k:                        Top-k hasil per metode retriever.
        hyde_num_hypothetical_docs: Jumlah hypothetical docs untuk HyDE.
        similarity_threshold:     Threshold untuk chunk bundling.
        max_neighbors:            Maksimal neighbor yang dicek per chunk.

    Returns:
        List of (Paragraph dengan bundled text, combined_score)
    """
    all_results: list[tuple[Paragraph, float, str]] = []

    # 1. Jalankan ketiga retriever secara paralel (atau berurutan jika executor tidak tersedia)
    print(f"[INFO] Running multi-path retrieval (BM25 + FAISS + HyDE)...")

    def _bm25_retrieve() -> list[tuple[Paragraph, float, str]]:
        try:
            topk = retrieve_topk(bm25, paragraphs, query, k=k)
            return [(para, score, "BM25") for para, score in topk]
        except Exception as e:
            print(f"[WARN] BM25 retrieval failed: {e}")
            return []

    def _faiss_retrieve() -> list[tuple[Paragraph, float, str]]:
        try:
            topk = retrieve_topk_faiss(embedding_model, faiss_index, paragraphs, query, k=k)
            return [(para, score, "FAISS") for para, score in topk]
        except Exception as e:
            print(f"[WARN] FAISS retrieval failed: {e}")
            return []

    def _hyde_retrieve() -> list[tuple[Paragraph, float, str]]:
        try:
            topk, _ = retrieve_topk_hyde(
                embedding_model,
                None,
                paragraphs,
                query,
                gen_pipeline=gen_pipeline,
                k=k,
                num_hypothetical_docs=hyde_num_hypothetical_docs,
            )
            return [(para, score, "HyDE") for para, score in topk]
        except Exception as e:
            print(f"[WARN] HyDE retrieval failed: {e}")
            return []

    # Try parallel execution first
    if ThreadPoolExecutor is not None:
        try:
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    executor.submit(_bm25_retrieve): "BM25",
                    executor.submit(_faiss_retrieve): "FAISS",
                    executor.submit(_hyde_retrieve): "HyDE",
                }
                for future in as_completed(futures):
                    try:
                        results = future.result()
                        all_results.extend(results)
                    except Exception as e:
                        print(f"[WARN] Parallel retrieval failed: {e}")
        except Exception as e:
            print(f"[WARN] Parallel execution not available, falling back to sequential: {e}")
            # Fallback ke sequential
            all_results.extend(_bm25_retrieve())
            all_results.extend(_faiss_retrieve())
            all_results.extend(_hyde_retrieve())
    else:
        # Fallback ke sequential jika ThreadPoolExecutor tidak tersedia
        all_results.extend(_bm25_retrieve())
        all_results.extend(_faiss_retrieve())
        all_results.extend(_hyde_retrieve())

    if not all_results:
        print("[WARN] No results from any retriever, returning empty list")
        return []

    print(f"[INFO] Collected {len(all_results)} results from multi-path retrieval")

    # 2. Deduplicate berdasarkan paragraph_id
    deduplicated = _deduplicate_results(all_results)
    print(f"[INFO] After deduplication: {len(deduplicated)} unique chunks")

    # 3. Chunk expansion/bundling dengan similarity threshold
    bundled = _expand_chunks_with_bundling(
        deduplicated,
        paragraphs,
        embedding_model,
        similarity_threshold=similarity_threshold,
        max_neighbors=max_neighbors,
    )
    print(f"[INFO] After chunk bundling: {len(bundled)} bundled chunks")

    # 4. Sort by score dan kembalikan top-k
    bundled.sort(key=lambda x: (-x[1], x[0].paragraph_id))
    topk = bundled[:k]

    return [(para, score) for para, score, _, _ in topk]