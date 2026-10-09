from __future__ import annotations

import re


try:
    from tqdm.auto import tqdm  # type: ignore[import-not-found]
except Exception:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s%$.,:/-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    text = normalize_text(text)
    return [t for t in re.split(r"\W+", text) if t]
