from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

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


_TABLE_MARKER_ATTR = "data-finder-table"
_TABLE_BULLETS = {"•", "·", "◦", "▪", "●", "-", "—"}


def table_to_rows(table: Any) -> list[str]:
    """
    Ubah <table> menjadi baris teks "sel | sel | sel".

    Tabel 10-K (inline XBRL) memecah simbol ke sel tersendiri, mis.
    ["$", "8,259", "52", "%"], jadi simbol digabung kembali ke angkanya.
    """
    rows: list[str] = []
    for tr in table.find_all("tr"):
        cells: list[str] = []
        pending_prefix = ""
        for cell in tr.find_all(["td", "th"]):
            txt = re.sub(r"\s+", " ", cell.get_text(" ", strip=True)).strip()
            if not txt or txt in _TABLE_BULLETS:
                continue
            if txt in {"$", "(", "($", "€", "£"}:
                pending_prefix += txt
                continue
            if txt in {"%", ")", "%)", ")%"} and cells:
                cells[-1] += txt
                continue
            cells.append(pending_prefix + txt)
            pending_prefix = ""
        if cells:
            rows.append(" | ".join(cells))
    return rows


def split_table_rows(rows: list[str], chunk_words: int = 180) -> list[str]:
    """Chunk tabel per baris; baris pertama (header) diulang di tiap chunk lanjutan."""
    if not rows:
        return []
    text = " ; ".join(rows)
    if len(text.split()) <= chunk_words:
        return [text]

    header = rows[0]
    header_words = len(header.split())
    chunks: list[str] = []
    current: list[str] = [header]
    current_words = header_words
    for row in rows[1:]:
        row_words = len(row.split()) + 1  # +1 untuk separator " ; "
        if current_words + row_words > chunk_words and len(current) > 1:
            chunks.append(" ; ".join(current))
            current = [header]
            current_words = header_words
        current.append(row)
        current_words += row_words
    if len(current) > 1:
        chunks.append(" ; ".join(current))

    # Baris tunggal yang sangat panjang tetap dipotong per kata.
    result: list[str] = []
    for chunk in chunks:
        result.extend(split_long_paragraph(chunk, chunk_words=chunk_words))
    return result


def table_looks_like_noise(text: str) -> bool:
    # Tabel keuangan didominasi angka, jadi cukup minta ada label teksnya.
    t = text.strip()
    return len(t) < 40 or sum(ch.isalpha() for ch in t) < 10


def parse_10k_html_to_paragraphs(html_path: Path, show_progress: bool = False) -> list[str]:
    if BeautifulSoup is None:
        raise RuntimeError("Missing dependency 'beautifulsoup4'.")

    html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    for tag_name in ["script", "style", "noscript", "svg", "img"]:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    for tag_name in ["ix:header", "ix:hidden", "xbrli:context", "xbrli:unit"]:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Tabel diganti placeholder <p> berisi barisnya agar urutan dokumen tetap
    # terjaga. Dibalik supaya tabel bersarang diproses lebih dulu.
    for table in reversed(soup.find_all("table")):
        rows = table_to_rows(table)
        placeholder = soup.new_tag("p")
        placeholder[_TABLE_MARKER_ATTR] = "1"
        placeholder.string = "\n".join(rows)
        table.replace_with(placeholder)

    body = soup.body or soup
    blocks = body.find_all(["p", "div", "li", "h1", "h2", "h3", "h4"])

    # (is_table, text)
    raw_paragraphs: list[tuple[bool, str]] = []
    for block in tqdm(
        blocks,
        desc=f"Parse blocks {html_path.name}",
        disable=not show_progress,
        leave=False,
    ):
        if block.get(_TABLE_MARKER_ATTR):
            rows = [r for r in block.get_text().split("\n") if r.strip()]
            txt = " ; ".join(rows)
            if table_looks_like_noise(txt):
                continue
            raw_paragraphs.append((True, "\n".join(rows)))
            continue

        # Wrapper yang membungkus tabel akan menduplikasi isi tabel.
        if block.find(attrs={_TABLE_MARKER_ATTR: "1"}) is not None:
            continue

        txt = block.get_text(" ", strip=True)
        txt = re.sub(r"\s+", " ", txt).strip()
        if looks_like_noise(txt):
            continue
        raw_paragraphs.append((False, txt))

    seen = set()
    paragraphs: list[str] = []
    for is_table, para in raw_paragraphs:
        norm = normalize_text(para)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        if is_table:
            paragraphs.extend(split_table_rows(para.split("\n")))
        else:
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
