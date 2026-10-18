from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import datetime
from collections import defaultdict
from typing import Any

import pandas as pd

from .company import CompanyIndex, load_or_build_company_names
from .config import AppConfig, validate_config
from .data import load_train_queries, stratified_sample
from .evaluation import (
    context_recall_heuristic,
    context_recall_ragas_llm,
    context_recall_ragas_nonllm,
)
from .preprocess import load_or_build_corpus
from .retrieval import (
    build_bm25,
    retrieve_topk,
    build_hyde_retriever,
    build_hyde_llm,
    retrieve_topk_hyde,
    build_faiss_retriever,
    retrieve_topk_faiss,
    retrieve_multipath,
)
from .utils import tqdm


def _fmt_score(score: float | None) -> str:
    return "N/A" if score is None or math.isnan(score) else f"{score:.2f}"


def markdown_table(
    columns: dict[str, tuple[dict[str, float], float]],
    categories: list[str],
) -> str:
    """columns: {judul kolom: (skor per kategori, skor total)}."""
    names = list(columns)
    lines = [
        "| Category | " + " | ".join(names) + " |",
        "|---|" + "---:|" * len(names),
    ]
    for cat in categories:
        cells = [_fmt_score(columns[n][0].get(cat)) for n in names]
        lines.append(f"| {cat} | " + " | ".join(cells) + " |")
    lines.append("| Total | " + " | ".join(_fmt_score(columns[n][1]) for n in names) + " |")
    return "\n".join(lines)


def eval_label(config: AppConfig) -> str:
    """Label evaluator; untuk ragas_llm menyertakan model judge karena skornya bergantung pada judge."""
    if config.eval_mode == "ragas_llm":
        return f"ragas_llm (judge: {config.openrouter_model})"
    return config.eval_mode


def _aggregate(
    per_category: dict[str, list[float]],
    all_scores: list[float],
    categories: list[str],
) -> tuple[dict[str, float], float]:
    agg = {
        cat: float(sum(per_category[cat]) / len(per_category[cat])) if per_category.get(cat) else float("nan")
        for cat in categories
    }
    total = float(sum(all_scores) / len(all_scores)) if all_scores else 0.0
    return agg, total


def run_pipeline(config: AppConfig) -> dict[str, Any]:
    validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    method_name_map = {
        "bm25": "BM25",
        "hyde": "HyDE",
        "faiss": "FAISS",
        "multipath": "MPR",
    }
    method_name = method_name_map[config.retrieval_method]
    method_name_lower = config.retrieval_method
    run_prefix = f"{method_name_lower}_context_recall_{timestamp}"
    fallback_experiment_name = f"experiment_{timestamp}"

    # Always write into a dedicated experiment folder.
    # If experiment_name is empty, create an automatic timestamped folder.
    experiment_name = config.experiment_name.strip() or fallback_experiment_name
    experiment_dir = config.output_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    markdown_out_path = experiment_dir / f"{run_prefix}.md"
    csv_out_path = experiment_dir / f"{run_prefix}.csv"
    retrieval_details_path = experiment_dir / f"{run_prefix}_retrieval_details.jsonl"
    config_out_path = experiment_dir / f"{run_prefix}_config.json"

    print(f"[INFO] Loading train data from: {config.train_dir}")
    df = load_train_queries(config.train_dir, show_progress=not config.no_progress)
    print(f"[INFO] Valid train queries before sampling: {len(df)}")

    if len(df) == 0:
        raise RuntimeError(
            "Valid train queries is 0 after filtering. Check [DIAG] logs above."
        )

    df = stratified_sample(df, sample_percent=config.sample_percent, seed=config.sample_seed)
    print(f"[INFO] Final train queries to evaluate: {len(df)}")

    print(f"[INFO] Building/loading corpus from: {config.tenk_dir}")
    paragraphs = load_or_build_corpus(
        config.tenk_dir,
        config.corpus_cache_path,
        show_progress=not config.no_progress,
    )
    print(f"[INFO] Total corpus paragraphs: {len(paragraphs)}")

    company_index = None
    if config.company_filter:
        company_names = load_or_build_company_names(
            config.tenk_dir,
            config.company_names_cache_path,
            show_progress=not config.no_progress,
        )
        company_index = CompanyIndex(paragraphs, company_names)
        print(
            f"[INFO] Company filter enabled: {len(company_index.indices)} companies, "
            f"{sum(1 for n in company_names.values() if n)} registrant names"
        )

    # Setup retriever based on method
    if config.retrieval_method == "hyde":
        print(f"[INFO] Building HyDE retriever with {config.hyde_llm_model}")
        embedding_model, paragraph_embeddings = build_hyde_retriever(
            paragraphs,
            model_name=config.embedding_model,
            device=config.hyde_device,
            cache_path=config.hyde_embeddings_cache_path,
        )
        print(f"[INFO] Loading LLM pipeline (device={config.hyde_device})")
        gen_pipeline = build_hyde_llm(
            llm_model_name=config.hyde_llm_model,
            device=config.hyde_device,
        )
        bm25 = None
        faiss_index = None
    elif config.retrieval_method == "faiss":
        print(f"[INFO] Building FAISS dense retriever with {config.embedding_model}")
        embedding_model, faiss_index, paragraph_embeddings = build_faiss_retriever(
            paragraphs,
            model_name=config.embedding_model,
            device=config.hyde_device,
            cache_path=config.faiss_embeddings_cache_path,
        )
        bm25 = None
        gen_pipeline = None
    elif config.retrieval_method == "multipath":
        print("[INFO] Building Multi-Path Retriever (BM25 + FAISS + HyDE)...")
        # Build BM25
        print(f"[INFO] Building BM25 index (k1={config.bm25_k1}, b={config.bm25_b})")
        bm25, _ = build_bm25(paragraphs, k1=config.bm25_k1, b=config.bm25_b)
        # Build FAISS
        print(f"[INFO] Building FAISS dense retriever with {config.embedding_model}")
        embedding_model, faiss_index, paragraph_embeddings = build_faiss_retriever(
            paragraphs,
            model_name=config.embedding_model,
            device=config.hyde_device,
            cache_path=config.faiss_embeddings_cache_path,
        )
        # Build HyDE
        print(f"[INFO] Loading LLM pipeline for HyDE (device={config.hyde_device})")
        gen_pipeline = build_hyde_llm(
            llm_model_name=config.hyde_llm_model,
            device=config.hyde_device,
        )
    else:
        print(f"[INFO] Building BM25 index (k1={config.bm25_k1}, b={config.bm25_b})")
        bm25, _ = build_bm25(paragraphs, k1=config.bm25_k1, b=config.bm25_b)
        embedding_model = None
        paragraph_embeddings = None
        gen_pipeline = None
        faiss_index = None

    per_category_scores: dict[str, list[float]] = defaultdict(list)
    all_scores: list[float] = []
    per_category_heuristic: dict[str, list[float]] = defaultdict(list)
    all_heuristic: list[float] = []
    retrieval_records: list[dict[str, Any]] = []

    print(f"[INFO] Evaluating context recall with mode: {config.eval_mode}")
    for i, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Evaluate queries",
        disable=config.no_progress,
    ):
        query = row["query"]
        references = row["references_list"]
        category = row["category_std"]

        detected_companies: list[str] = []
        candidate_indices = None
        if company_index is not None:
            detected_companies = company_index.detect(query)
            candidate_indices = company_index.candidate_indices(detected_companies)

        if config.retrieval_method == "hyde":
            topk, hypothetical_docs = retrieve_topk_hyde(
                embedding_model,
                paragraph_embeddings,
                paragraphs,
                query,
                gen_pipeline=gen_pipeline,
                k=config.top_k,
                num_hypothetical_docs=config.hyde_num_hypothetical_docs,
                candidate_indices=candidate_indices,
            )
        elif config.retrieval_method == "faiss":
            topk = retrieve_topk_faiss(
                embedding_model,
                faiss_index,
                paragraphs,
                query,
                k=config.top_k,
                paragraph_embeddings=paragraph_embeddings,
                candidate_indices=candidate_indices,
            )
            hypothetical_docs = None
        elif config.retrieval_method == "multipath":
            topk, hypothetical_docs = retrieve_multipath(
                bm25,
                embedding_model,
                faiss_index,
                paragraph_embeddings,
                paragraphs,
                query,
                gen_pipeline=gen_pipeline,
                k=config.top_k,
                hyde_num_hypothetical_docs=config.hyde_num_hypothetical_docs,
                similarity_threshold=config.chunk_expansion_similarity_threshold,
                max_neighbors=config.chunk_expansion_max_neighbors,
                candidate_indices=candidate_indices,
                candidates_per_retriever=config.multipath_candidates_per_retriever,
                rrf_k=config.rrf_k,
            )
        else:
            topk = retrieve_topk(
                bm25,
                paragraphs,
                query,
                k=config.top_k,
                candidate_indices=candidate_indices,
            )
            hypothetical_docs = None

        retrieved_texts = [p.text for p, _ in topk]

        # Heuristic selalu dihitung sebagai pembanding deterministik
        # (skor ragas_llm bisa berubah drastis antar model judge).
        heuristic_score = context_recall_heuristic(
            references,
            retrieved_texts,
            fuzzy_threshold=config.fuzzy_threshold,
        )
        per_category_heuristic[category].append(heuristic_score)
        all_heuristic.append(heuristic_score)

        if config.eval_mode == "heuristic":
            score = heuristic_score
        elif config.eval_mode == "ragas_nonllm":
            try:
                score = context_recall_ragas_nonllm(references, retrieved_texts)
            except Exception as exc:
                print(
                    f"[ERROR] NonLLMContextRecall failed at row {i}. Error: {exc}"
                )
                raise
        else:
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    score = context_recall_ragas_llm(
                        query=query,
                        references=references,
                        retrieved_texts=retrieved_texts,
                        api_key=config.openrouter_api_key,
                        endpoint=config.openrouter_endpoint,
                        model=config.openrouter_model,
                    )
                    break
                except Exception as exc:
                    if attempt < max_retries:
                        print(
                            f"[WARN] RAGAS/OpenRouter failed at row {i}, "
                            f"attempt {attempt}/{max_retries}. Retrying... Error: {exc}"
                        )
                        import time
                        time.sleep(2)
                    else:
                        print(
                            f"[ERROR] RAGAS/OpenRouter failed at row {i} "
                            f"after {max_retries} attempts. Error: {exc}"
                        )
                        raise

        score = max(0.0, min(100.0, float(score)))
        per_category_scores[category].append(score)
        all_scores.append(score)

        record = {
            "query_index": int(i),
            "query": query,
            "category": category,
            "score": score,
            "heuristic_score": heuristic_score,
            "references": references,
            "detected_companies": detected_companies,
            "company_filter_applied": candidate_indices is not None,
            "topk": [
                {
                    "paragraph_id": p.paragraph_id,
                    "source_file": p.source_file,
                    "score": s,
                    "text": p.text,
                }
                for p, s in topk
            ],
        }
        if hypothetical_docs is not None:
            record["hypothetical_documents"] = hypothetical_docs
        retrieval_records.append(record)

    agg, total = _aggregate(per_category_scores, all_scores, config.target_categories)
    heuristic_agg, heuristic_total = _aggregate(per_category_heuristic, all_heuristic, config.target_categories)

    n_filtered = sum(1 for r in retrieval_records if r["company_filter_applied"])
    if company_index is not None:
        print(f"[INFO] Company filter applied to {n_filtered}/{len(retrieval_records)} queries")

    columns = {f"{method_name} (Context Recall)": (agg, total)}
    if config.eval_mode != "heuristic":
        columns[f"{method_name} (Heuristic)"] = (heuristic_agg, heuristic_total)
    md_table = markdown_table(columns, config.target_categories)
    md_header = "\n".join([
        f"- Evaluator: {eval_label(config)}",
        f"- Company filter: {'on (ablation, applied to ' + str(n_filtered) + '/' + str(len(retrieval_records)) + ' queries)' if config.company_filter else 'off (full corpus)'}",
        f"- Corpus: {config.corpus_cache_path.name}",
        f"- Queries: {len(retrieval_records)} (sample {config.sample_percent}%, seed {config.sample_seed}), top-k {config.top_k}",
    ])
    print(f"\n=== {method_name} Context Recall (0-100) ===")
    print(md_header)
    print(md_table)

    markdown_out_path.write_text(md_header + "\n\n" + md_table + "\n", encoding="utf-8")

    rows = []
    for cat in config.target_categories + ["Total"]:
        row: dict[str, Any] = {
            "Category": cat,
            method_name: total if cat == "Total" else agg[cat],
        }
        if config.eval_mode != "heuristic":
            row[f"{method_name} (heuristic)"] = heuristic_total if cat == "Total" else heuristic_agg[cat]
        rows.append(row)
    pd.DataFrame(rows).to_csv(csv_out_path, index=False)

    with retrieval_details_path.open("w", encoding="utf-8") as f:
        for rec in retrieval_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    config_payload = asdict(config)
    # Jangan tulis API key ke folder outputs (ikut ter-commit ke git).
    config_payload["openrouter_api_key"] = "<redacted>" if config.openrouter_api_key else ""
    config_payload["eval_label"] = eval_label(config)
    config_payload["root"] = str(config.root)
    config_payload["train_dir"] = str(config.train_dir)
    config_payload["tenk_dir"] = str(config.tenk_dir)
    config_payload["output_dir"] = str(config.output_dir)
    config_payload["corpus_cache_path"] = str(config.corpus_cache_path)
    config_payload["timestamp"] = timestamp
    config_payload["experiment_dir"] = str(experiment_dir)
    config_payload["experiment_name_resolved"] = experiment_name
    config_payload["output_files"] = {
        "markdown": str(markdown_out_path),
        "csv": str(csv_out_path),
        "retrieval_details": str(retrieval_details_path),
    }
    with config_out_path.open("w", encoding="utf-8") as f:
        json.dump(config_payload, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] Saved markdown table: {markdown_out_path}")
    print(f"[INFO] Saved csv summary: {csv_out_path}")
    print(f"[INFO] Saved retrieval details: {retrieval_details_path}")
    print(f"[INFO] Saved run config: {config_out_path}")

    return {
        "method": method_name,
        "eval_label": eval_label(config),
        "category_scores": agg,
        "total": total,
        "heuristic_category_scores": heuristic_agg,
        "heuristic_total": heuristic_total,
        "experiment_dir": experiment_dir,
    }
