from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from .preprocess import Paragraph, extract_dei_value
from .utils import tqdm


# Ticker yang juga kata umum atau singkatan keuangan di query
# (mis. "IT capex", "CF", "NI", "GM"). Ticker ini, dan semua ticker satu huruf,
# hanya diterima jika muncul dalam pola eksplisit, mis. "(ON)", "EA ticker",
# "MCK's", "F: ...".
_AMBIGUOUS_WORDS = {
    "ALL", "ARE", "KEY", "LOW", "NOW", "DAY", "CAT", "HAS", "WELL", "BALL",
    "IT", "CF", "NI", "IP", "MA", "PM", "IR", "GM", "DD", "FI", "CI", "SO",
    "ES", "RF", "GL", "ON", "DE", "EL", "CE", "TT", "BR", "LH",
}

# Kata pertama nama perusahaan yang terlalu generik untuk dipakai sebagai alias.
_GENERIC_FIRST_WORDS = {
    "the", "general", "american", "united", "first", "national", "international",
    "global", "public", "southern", "western", "eastern", "northern", "digital",
    "consolidated", "federal", "universal", "applied", "advanced", "texas",
    "republic", "regions", "principal", "marathon", "pioneer", "eli",
    "capital", "best", "live", "match", "progressive", "intuitive", "host",
    "steel", "waste", "citizens", "fifth", "state", "realty", "air", "old",
    "news", "equity", "energy",
}

# Alias nama yang juga kata Inggris biasa; di awal query (huruf kapital karena
# awal kalimat) wajib diikuti "'s" atau sufiks badan usaha.
_COMMON_WORD_ALIASES = {"booking", "ball", "pool", "target", "dover", "mosaic"}

_NAME_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "company", "co", "ltd",
    "limited", "plc", "holdings", "holding", "llc", "lp", "nv", "sa", "ag",
    "se", "group", "the", "companies",
}
_SUFFIX_PATTERN = r"(?:inc|incorporated|corp|corporation|company|co|ltd|plc|holdings|group)\b"

def extract_registrant_name(html_path: Path) -> str:
    """Ambil nama perusahaan dari tag inline XBRL dei:EntityRegistrantName."""
    with html_path.open("r", encoding="utf-8", errors="ignore") as f:
        head = f.read(3_000_000)
    return extract_dei_value(head, "EntityRegistrantName") or extract_dei_value(
        html_path.read_text(encoding="utf-8", errors="ignore"), "EntityRegistrantName"
    )


def load_or_build_company_names(
    tenk_dir: Path,
    cache_path: Path,
    show_progress: bool = True,
) -> dict[str, str]:
    """Mapping ticker (nama file 10-K) -> nama registrant. Di-cache ke JSON."""
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    html_files = sorted(tenk_dir.glob("*.html"))
    if not html_files:
        print(f"[WARN] No HTML files in {tenk_dir}; company filter will use tickers only")
        return {}

    names: dict[str, str] = {}
    for html_file in tqdm(html_files, desc="Extract company names", disable=not show_progress):
        names[html_file.stem] = extract_registrant_name(html_file)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(names, f, ensure_ascii=False, indent=2)
    return names


def _core_name_tokens(name: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", name.lower().replace("&", " and "))
    while tokens and tokens[0] == "the":
        tokens = tokens[1:]
    while tokens and tokens[-1] in _NAME_SUFFIXES:
        tokens = tokens[:-1]
    return tokens


class CompanyIndex:
    """
    Deteksi perusahaan target dari query (ticker atau nama) dan kembalikan
    indeks paragraf korpus milik perusahaan tersebut.
    """

    def __init__(self, paragraphs: Sequence[Paragraph], company_names: dict[str, str] | None = None):
        by_ticker: dict[str, list[int]] = defaultdict(list)
        for idx, para in enumerate(paragraphs):
            by_ticker[Path(para.source_file).stem.upper()].append(idx)
        self.indices = {t: np.asarray(v, dtype=np.int64) for t, v in by_ticker.items()}

        # Alias ticker: "BRK-B" juga dikenali sebagai "BRK.B" dan "BRK".
        self.ticker_aliases: dict[str, str] = {}
        base_counts: dict[str, int] = defaultdict(int)
        for ticker in self.indices:
            base_counts[re.split(r"[-.]", ticker)[0]] += 1
        for ticker in self.indices:
            self.ticker_aliases[ticker] = ticker
            self.ticker_aliases[ticker.replace("-", ".")] = ticker
            base = re.split(r"[-.]", ticker)[0]
            if base != ticker and base_counts[base] == 1 and base not in self.indices:
                self.ticker_aliases[base] = ticker

        # Nama lengkap (multi-kata) dan alias kata pertama yang unik.
        self.full_names: list[tuple[re.Pattern[str], str]] = []
        self.first_word_alias: dict[str, str] = {}
        first_word_counts: dict[str, int] = defaultdict(int)
        cores: dict[str, list[str]] = {}
        for ticker, name in (company_names or {}).items():
            ticker = ticker.upper()
            if ticker not in self.indices or not name:
                continue
            core = _core_name_tokens(name)
            if not core:
                continue
            cores[ticker] = core
            first_word_counts[core[0]] += 1
        for ticker, core in cores.items():
            parts = [r"(?:and|&)" if t == "and" else re.escape(t) for t in core]
            pattern = r"\b" + r"[\s\-.,']*".join(parts) + r"\b"
            if len(core) == 1:
                # Nama satu kata (mis. "News Corp") wajib diikuti sufiks badan usaha.
                pattern += r"[\s,.]+" + _SUFFIX_PATTERN
            self.full_names.append((re.compile(pattern, re.IGNORECASE), ticker))
            first = core[0]
            if (
                first_word_counts[first] == 1
                and len(first) >= 4
                and first not in _GENERIC_FIRST_WORDS
            ):
                self.first_word_alias[first] = ticker

    def detect(self, query: str) -> list[str]:
        """Kembalikan daftar ticker yang disebut dalam query (urut kemunculan)."""
        found: list[str] = []

        def add(ticker: str) -> None:
            if ticker not in found:
                found.append(ticker)

        for pattern, ticker in self.full_names:
            if pattern.search(query):
                add(ticker)

        tokens = list(re.finditer(r"[A-Za-z][A-Za-z0-9.\-]*", query))
        for pos, m in enumerate(tokens):
            raw = m.group(0).rstrip(".-")
            start, end = m.start(), m.start() + len(raw)
            before = query[max(0, start - 1):start]
            after = query[end:end + 2]

            # Alias nama: kata pertama nama perusahaan, wajib diawali huruf kapital.
            word = re.sub(r"[^A-Za-z]", "", raw).lower()
            if raw[:1].isupper() and word in self.first_word_alias:
                next_raw = tokens[pos + 1].group(0).lower().rstrip(".") if pos + 1 < len(tokens) else ""
                if (
                    pos > 0
                    or word not in _COMMON_WORD_ALIASES
                    or after[:2] in {"'s", "’s", "�s"}
                    or next_raw in _NAME_SUFFIXES
                ):
                    add(self.first_word_alias[word])
                    continue

            ticker = self.ticker_aliases.get(raw)
            if ticker is None:
                continue
            # Bagian dari singkatan seperti D&A, R&D, D/E, T&D, SG&A,
            # atau satuan angka seperti "25.2K".
            if before in {"&", "/"} or after[:1] in {"&", "/"} or before.isdigit() or before == ".":
                continue

            ambiguous = len(raw) == 1 or raw in _AMBIGUOUS_WORDS
            if not ambiguous:
                add(ticker)
                continue

            prev_word = tokens[pos - 1].group(0).lower() if pos > 0 else ""
            next_word = tokens[pos + 1].group(0).lower() if pos + 1 < len(tokens) else ""
            explicit = (
                (before == "(" and after[:1] == ")")
                or after[:2] in {"'s", "’s", "�s"}
                or prev_word in {"ticker", "tkr"}
                or next_word == "ticker"
                or pos == len(tokens) - 1
                # Ticker di awal query (bukan "A" yang bisa berupa artikel).
                or (pos == 0 and raw != "A")
            )
            if explicit:
                add(ticker)

        return found

    def candidate_indices(self, tickers: Sequence[str]) -> np.ndarray | None:
        """Indeks paragraf milik ticker yang terdeteksi; None berarti cari global."""
        parts = [self.indices[t] for t in tickers if t in self.indices]
        if not parts:
            return None
        return np.sort(np.concatenate(parts))
