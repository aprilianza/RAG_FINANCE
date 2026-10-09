from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

try:
    from bs4 import BeautifulSoup  # type: ignore[import-not-found]
except Exception:
    BeautifulSoup = None

from .utils import normalize_text, tqdm


@dataclass
class Paragraph:
    paragraph_id: str
    source_file: str
    text: str


def looks_like_noise(text: str) -> bool:
    if not text:
        return True

    t = text.strip()
    if len(t) < 40:
        return True

    alpha = sum(ch.isalpha() for ch in t)
    digit = sum(ch.isdigit() for ch in t)
    if alpha == 0:
        return True

    if digit > alpha * 2:
        return True

    upper_ratio = sum(ch.isupper() for ch in t if ch.isalpha()) / max(alpha, 1)
    if upper_ratio > 0.85 and len(t.split()) < 12:
        return True

    return False


def split_long_paragraph(text: str, chunk_words: int = 180, overlap_words: int = 30) -> list[str]:
    words = text.split()
    if len(words) <= chunk_words:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start = max(end - overlap_words, start + 1)

    return chunks


def parse_10k_html_to_paragraphs(html_path: Path, show_progress: bool = False) -> list[str]:
    if BeautifulSoup is None:
        raise RuntimeError("Missing dependency 'beautifulsoup4'.")

    html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    for tag_name in ["script", "style", "noscript", "svg", "img", "table"]:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    for tag_name in ["ix:header", "ix:hidden", "xbrli:context", "xbrli:unit"]:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    body = soup.body or soup
    blocks = body.find_all(["p", "div", "li", "h1", "h2", "h3", "h4"])

    raw_paragraphs: list[str] = []
    for block in tqdm(
        blocks,
        desc=f"Parse blocks {html_path.name}",
        disable=not show_progress,
        leave=False,
    ):
        txt = block.get_text(" ", strip=True)
        txt = re.sub(r"\s+", " ", txt).strip()
        if looks_like_noise(txt):
            continue
        raw_paragraphs.append(txt)

    seen = set()
    paragraphs: list[str] = []
    for para in raw_paragraphs:
        norm = normalize_text(para)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        paragraphs.extend(split_long_paragraph(para))

    return paragraphs


def load_or_build_corpus(tenk_dir: Path, cache_path: Path, show_progress: bool = True) -> List[Paragraph]:
    if cache_path.exists():
        paragraphs: list[Paragraph] = []
        with cache_path.open("r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Load cached corpus", disable=not show_progress):
                obj = json.loads(line)
                paragraphs.append(
                    Paragraph(
                        paragraph_id=obj["paragraph_id"],
                        source_file=obj["source_file"],
                        text=obj["text"],
                    )
                )
        return paragraphs

    html_files = sorted(tenk_dir.glob("*.html"))
    if not html_files:
        raise FileNotFoundError(f"No HTML files found in {tenk_dir}")

    paragraphs: list[Paragraph] = []
    paragraph_counter = 0

    for html_file in tqdm(html_files, desc="Parse 10-K files", disable=not show_progress):
        parsed = parse_10k_html_to_paragraphs(html_file, show_progress=False)
        for text in parsed:
            paragraph_counter += 1
            paragraphs.append(
                Paragraph(
                    paragraph_id=f"p{paragraph_counter}",
                    source_file=html_file.name,
                    text=text,
                )
            )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        for p in tqdm(paragraphs, desc="Write corpus cache", disable=not show_progress):
            f.write(
                json.dumps(
                    {
                        "paragraph_id": p.paragraph_id,
                        "source_file": p.source_file,
                        "text": p.text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return paragraphs
