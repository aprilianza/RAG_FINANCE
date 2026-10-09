from __future__ import annotations

import ast
import re
from collections.abc import Sequence as ABCSequence
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .utils import tqdm


def map_category(raw_category: str) -> Optional[str]:
    if not isinstance(raw_category, str):
        return None

    c = raw_category.strip().lower()
    if not c:
        return None

    c = re.sub(r"[_\-/]+", " ", c)
    c = re.sub(r"\s+", " ", c).strip()

    mapping = {
        "accounting": "Accounting",
        "company overview": "Company Overview",
        "overview": "Company Overview",
        "financials": "Financials",
        "financial": "Financials",
        "footnotes": "Footnotes",
        "governance": "Governance",
        "legal": "Legal",
        "risk": "Risk",
        "shareholder return": "Shareholder Return",
        "shareholder returns": "Shareholder Return",
    }

    if c in mapping:
        return mapping[c]

    for key, value in mapping.items():
        if key in c:
            return value

    return None


def clean_references(x: Any) -> list[str]:
    if x is None:
        return []

    try:
        if pd.isna(x):
            return []
    except Exception:
        pass

    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    return [str(r).strip() for r in parsed if str(r).strip()]
            except Exception:
                pass
        return [s]

    if hasattr(x, "tolist"):
        try:
            x = x.tolist()
        except Exception:
            pass

    if isinstance(x, ABCSequence) and not isinstance(x, (str, bytes, bytearray)):
        return [str(r).strip() for r in x if str(r).strip()]

    return []


def load_train_queries(train_dir: Path, show_progress: bool = True) -> pd.DataFrame:
    parquet_files = sorted(train_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {train_dir}")

    frames = []
    for pf in tqdm(parquet_files, desc="Load parquet", disable=not show_progress):
        frames.append(pd.read_parquet(pf))
    df = pd.concat(frames, ignore_index=True)

    required_cols = ["text", "answer", "references", "category"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in train parquet: {missing}")

    df = df.copy()
    df["query"] = df["text"].astype(str)
    df["category_std"] = df["category"].apply(map_category)

    total_rows = len(df)
    print(f"[DIAG] Total rows loaded: {total_rows}")
    print(
        "[DIAG] Top raw category labels (max 12): "
        f"{df['category'].astype(str).value_counts(dropna=False).head(12).to_dict()}"
    )

    unmatched_mask = df["category_std"].isna()
    unmatched_count = int(unmatched_mask.sum())
    if unmatched_count:
        print(f"[DIAG] Unmapped category rows: {unmatched_count}")
        print(
            "[DIAG] Unmapped category samples (max 12): "
            f"{df.loc[unmatched_mask, 'category'].astype(str).value_counts().head(12).to_dict()}"
        )

    non_empty_query_mask = df["query"].str.strip() != ""
    references_notna_mask = df["references"].notna()
    mapped_category_mask = df["category_std"].notna()

    print(f"[DIAG] Rows with non-empty query: {int(non_empty_query_mask.sum())}")
    print(f"[DIAG] Rows with references notna: {int(references_notna_mask.sum())}")
    print(f"[DIAG] Rows with mapped category: {int(mapped_category_mask.sum())}")

    df = df[non_empty_query_mask & references_notna_mask & mapped_category_mask]
    df["references_list"] = df["references"].apply(clean_references)

    non_empty_refs_mask = df["references_list"].map(len) > 0
    print(f"[DIAG] Rows with non-empty references_list: {int(non_empty_refs_mask.sum())}")
    df = df[non_empty_refs_mask]

    return df.reset_index(drop=True)


def stratified_sample(df: pd.DataFrame, sample_percent: float, seed: int) -> pd.DataFrame:
    import math

    if sample_percent >= 100.0:
        # shuffle for randomness but keep full set
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    frac = sample_percent / 100.0
    total = len(df)
    target_total = max(1, int(round(total * frac)))

    group_sizes = df.groupby("category_std", sort=False).size().to_dict()
    categories = list(group_sizes.keys())

    # Exact fractional allocations
    exact_alloc = {cat: group_sizes[cat] * frac for cat in categories}
    floor_alloc = {cat: int(math.floor(exact_alloc[cat])) for cat in categories}
    allocated = dict(floor_alloc)

    current_sum = sum(allocated.values())
    remainder = target_total - current_sum

    # Distribute remainder by largest fractional parts
    if remainder > 0:
        frac_parts = sorted(
            ((cat, exact_alloc[cat] - floor_alloc[cat]) for cat in categories),
            key=lambda x: (-x[1], -group_sizes[x[0]]),
        )
        for cat, _ in frac_parts:
            if remainder <= 0:
                break
            if allocated[cat] < group_sizes[cat]:
                allocated[cat] += 1
                remainder -= 1

    # If still remainder due to exhausted groups, distribute to any with capacity
    if remainder > 0:
        for cat in sorted(categories, key=lambda c: -group_sizes[c]):
            if remainder <= 0:
                break
            avail = group_sizes[cat] - allocated[cat]
            if avail <= 0:
                continue
            take = min(avail, remainder)
            allocated[cat] += take
            remainder -= take

    # Ensure minimum one per category if target allows
    non_empty_cats = [c for c in categories if group_sizes[c] > 0]
    if target_total >= len(non_empty_cats):
        need_fix = [c for c in non_empty_cats if allocated.get(c, 0) == 0]
        for c in need_fix:
            allocated[c] = 1

        # adjust down to match target_total if we overshot
        current_sum = sum(allocated.values())
        if current_sum > target_total:
            # reduce from largest allocations (excluding those just set to 1)
            reducible = sorted(
                [c for c in categories if allocated[c] > 1],
                key=lambda c: allocated[c],
                reverse=True,
            )
            idx = 0
            while current_sum > target_total and idx < len(reducible):
                c = reducible[idx]
                can_reduce = allocated[c] - 1
                reduce_by = min(can_reduce, current_sum - target_total)
                allocated[c] -= reduce_by
                current_sum -= reduce_by
                idx += 1

    # Final clamp to group sizes
    for c in categories:
        if allocated[c] > group_sizes[c]:
            allocated[c] = group_sizes[c]

    # Sample per group
    sampled_parts = []
    for cat, grp in df.groupby("category_std", sort=False):
        n = allocated.get(cat, 0)
        if n <= 0:
            continue
        n = min(n, len(grp))
        sampled_parts.append(grp.sample(n=n, random_state=seed))

    if not sampled_parts:
        return df.sample(n=1, random_state=seed).reset_index(drop=True)

    sampled = pd.concat(sampled_parts, ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    print(f"[INFO] Stratified sample enabled: {sample_percent:.2f}% -> {len(sampled)}/{len(df)} rows")
    print(f"[INFO] Allocation per category: {allocated}")
    return sampled
