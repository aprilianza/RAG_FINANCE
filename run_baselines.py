#!/usr/bin/env python3
"""
Jalankan ulang semua metode retrieval dengan konfigurasi, korpus, sampel, dan
judge yang sama, lalu tulis satu tabel perbandingan.

Contoh:
    python run_baselines.py --experiment-name baselines_v2
    python run_baselines.py --methods bm25 faiss --eval-mode heuristic
"""
from __future__ import annotations

import gc
from datetime import datetime

import pandas as pd

from finder_bm25 import run_pipeline
from finder_bm25.pipeline import markdown_table
from run_finder_bm25 import build_config, build_parser

ALL_METHODS = ["bm25", "faiss", "hyde", "multipath"]


def _free_gpu_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def main() -> None:
    parser = build_parser()
    parser.description = "Run all retrieval methods with identical settings and compare them"
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=ALL_METHODS)
    args = parser.parse_args()
    if args.retrieval_method is not None:
        parser.error("use --methods instead of --retrieval-method")

    base = build_config(args)
    if not base.experiment_name.strip():
        base.experiment_name = f"baselines_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    results = []
    for method in args.methods:
        config = build_config(args)
        config.experiment_name = base.experiment_name
        config.retrieval_method = method
        print(f"\n{'=' * 70}\n[RUN] {method} -> outputs/{config.experiment_name}\n{'=' * 70}")
        results.append(run_pipeline(config))
        _free_gpu_memory()

    categories = base.target_categories
    columns = {r["method"]: (r["category_scores"], r["total"]) for r in results}
    heuristic_columns = {r["method"]: (r["heuristic_category_scores"], r["heuristic_total"]) for r in results}

    sections = [
        f"# Baseline comparison: {base.experiment_name}",
        "",
        f"- Evaluator: {results[0]['eval_label']}",
        f"- Company filter: {'on (ablation)' if base.company_filter else 'off (full corpus)'}",
        f"- Corpus: {base.corpus_cache_path.name}",
        f"- Sample: {base.sample_percent}% (seed {base.sample_seed}), top-k {base.top_k}",
        "",
        f"## Context Recall ({results[0]['eval_label']})",
        "",
        markdown_table(columns, categories),
    ]
    if base.eval_mode != "heuristic":
        sections += ["", "## Context Recall (heuristic, deterministic)", "", markdown_table(heuristic_columns, categories)]
    report = "\n".join(sections) + "\n"

    out_dir = base.output_dir / base.experiment_name
    (out_dir / "comparison.md").write_text(report, encoding="utf-8")

    rows = []
    for cat in categories + ["Total"]:
        row = {"Category": cat}
        for r in results:
            row[r["method"]] = r["total"] if cat == "Total" else r["category_scores"][cat]
            if base.eval_mode != "heuristic":
                row[f"{r['method']} (heuristic)"] = (
                    r["heuristic_total"] if cat == "Total" else r["heuristic_category_scores"][cat]
                )
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "comparison.csv", index=False)

    print("\n" + report)
    print(f"[INFO] Saved comparison: {out_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
