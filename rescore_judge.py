#!/usr/bin/env python3
"""
Isi ulang skor judge (ragas_llm / ragas_nonllm) yang hilang atau tidak valid
dari file _retrieval_details.jsonl, tanpa mengulang retrieval maupun HyDE.

Baris yang dinilai ulang:
  - skor None (judge gagal / dilewati, run versi baru), dan/atau
  - --from-index N : semua baris mulai posisi N (0-based, urutan di file),
                     untuk run lama yang mengubah kegagalan judge jadi skor 100,
  - --scores 100   : baris dengan skor persis sama dengan nilai tersebut.

Contoh:
    python rescore_judge.py outputs/baselines_20261018_155709 --dry-run
    python rescore_judge.py outputs/baselines_20261018_155709 --methods multipath --from-index 346
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
from pathlib import Path

from finder_bm25.config import AppConfig
from finder_bm25.pipeline import (
    MAX_CONSECUTIVE_JUDGE_FAILURES,
    METHOD_NAMES,
    score_with_judge,
    summarize_records,
    write_run_summary,
)
from run_baselines import write_comparison

SAVE_EVERY = 10


def load_config(config_path: Path) -> AppConfig:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    names = {f.name for f in dataclasses.fields(AppConfig)}
    kwargs = {k: v for k, v in payload.items() if k in names and k != "openrouter_api_key"}
    kwargs["root"] = Path(kwargs.get("root", "."))
    return AppConfig(**kwargs)  # API key diambil dari env / .env


def load_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def find_runs(experiment_dir: Path, methods: list[str] | None) -> list[tuple[Path, AppConfig]]:
    """Run terbaru per metode di folder eksperimen."""
    latest: dict[str, tuple[Path, AppConfig]] = {}
    for config_path in sorted(experiment_dir.glob("*_config.json")):
        config = load_config(config_path)
        if methods and config.retrieval_method not in methods:
            continue
        latest[config.retrieval_method] = (config_path, config)
    return list(latest.values())


def select_rows(records: list[dict], from_index: int | None, scores: list[float]) -> list[int]:
    selected = []
    for pos, rec in enumerate(records):
        if (
            rec.get("score") is None
            or (from_index is not None and pos >= from_index)
            or (rec.get("score") is not None and float(rec["score"]) in scores)
        ):
            selected.append(pos)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-run the judge for missing/invalid scores in an experiment folder")
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--methods", nargs="+", choices=list(METHOD_NAMES), default=None)
    parser.add_argument("--from-index", type=int, default=None, help="Also rescore every row from this 0-based position")
    parser.add_argument("--scores", type=float, nargs="+", default=[], help="Also rescore rows whose score equals one of these values")
    parser.add_argument("--openrouter-api-key", type=str, default=None)
    parser.add_argument("--openrouter-model", type=str, default=None, help="Override the judge (results are then not comparable with other runs)")
    parser.add_argument("--dry-run", action="store_true", help="Only show which rows would be rescored")
    args = parser.parse_args()

    runs = find_runs(args.experiment_dir, args.methods)
    if not runs:
        parser.error(f"no *_config.json found in {args.experiment_dir}")

    stopped = False
    for config_path, config in runs:
        if config.eval_mode == "heuristic":
            print(f"[SKIP] {config_path.name}: eval_mode=heuristic has no judge scores")
            continue
        if args.openrouter_api_key:
            config.openrouter_api_key = args.openrouter_api_key
        if args.openrouter_model and args.openrouter_model != config.openrouter_model:
            print(f"[WARN] judge override {config.openrouter_model} -> {args.openrouter_model}; "
                  "rescored rows will not be comparable with the rest of the run")
            config.openrouter_model = args.openrouter_model

        prefix = config_path.name[: -len("_config.json")]
        details_path = args.experiment_dir / f"{prefix}_retrieval_details.jsonl"
        records = load_records(details_path)
        rows = select_rows(records, args.from_index, args.scores)
        print(f"[INFO] {prefix}: {len(rows)}/{len(records)} rows to rescore with {config.openrouter_model}")
        if args.dry_run or not rows:
            continue

        backup = details_path.with_suffix(".jsonl.bak")
        if not backup.exists():
            shutil.copy2(details_path, backup)
            print(f"[INFO] Backup: {backup}")

        method_name = METHOD_NAMES[config.retrieval_method]
        paths = (
            args.experiment_dir / f"{prefix}.md",
            args.experiment_dir / f"{prefix}.csv",
            details_path,
        )
        consecutive_failures = 0
        done = 0
        for n, pos in enumerate(rows, start=1):
            rec = records[pos]
            try:
                rec["score"] = score_with_judge(
                    config,
                    rec["query"],
                    rec["references"],
                    [t["text"] for t in rec["topk"]],
                )
                rec.pop("eval_error", None)
                consecutive_failures = 0
                done += 1
            except Exception as exc:
                rec["score"] = None
                rec["eval_error"] = f"{type(exc).__name__}: {exc}"
                consecutive_failures += 1
                print(f"[WARN] row {pos} failed: {rec['eval_error']}")
            print(f"[INFO] {n}/{len(rows)} row {pos} -> {rec['score']}")

            if consecutive_failures >= MAX_CONSECUTIVE_JUDGE_FAILURES:
                print(f"[ERROR] judge failed {consecutive_failures} times in a row; stopping. "
                      "Progress is saved, rerun this command later to continue.")
                stopped = True
                break
            if n % SAVE_EVERY == 0:
                with details_path.open("w", encoding="utf-8") as f:
                    for r in records:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"[INFO] {prefix}: rescored {done}/{len(rows)} rows")
        write_run_summary(config, records, method_name, *paths)
        if stopped:
            break

    # Perbarui comparison.md jika folder ini hasil run_baselines.py
    if not args.dry_run and (args.experiment_dir / "comparison.md").exists():
        all_runs = find_runs(args.experiment_dir, None)
        order = list(METHOD_NAMES)
        all_runs.sort(key=lambda r: order.index(r[1].retrieval_method))
        results = []
        for config_path, config in all_runs:
            prefix = config_path.name[: -len("_config.json")]
            records = load_records(args.experiment_dir / f"{prefix}_retrieval_details.jsonl")
            results.append(summarize_records(config, records, METHOD_NAMES[config.retrieval_method]))
        write_comparison(all_runs[0][1], results, args.experiment_dir)


if __name__ == "__main__":
    main()
