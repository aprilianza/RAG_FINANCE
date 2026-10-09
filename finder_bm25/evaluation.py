from __future__ import annotations

import difflib
from typing import Sequence

from .utils import normalize_text, tokenize


def sequence_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def jaccard_tokens(a: str, b: str) -> float:
    sa = set(tokenize(a))
    sb = set(tokenize(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def reference_hit(reference: str, retrieved_text: str, fuzzy_threshold: float = 0.85) -> bool:
    r = normalize_text(reference)
    t = normalize_text(retrieved_text)

    if not r or not t:
        return False

    if r in t or t in r:
        return True

    if sequence_similarity(r, t) >= fuzzy_threshold:
        return True

    if jaccard_tokens(r, t) >= 0.6:
        return True

    return False


def context_recall_heuristic(
    references: Sequence[str],
    retrieved_texts: Sequence[str],
    fuzzy_threshold: float = 0.85,
) -> float:
    refs = [r for r in references if str(r).strip()]
    if not refs:
        return 0.0

    hit_count = 0
    for ref in refs:
        if any(reference_hit(ref, ctx, fuzzy_threshold=fuzzy_threshold) for ctx in retrieved_texts):
            hit_count += 1

    return (hit_count / len(refs)) * 100.0


def context_recall_ragas_llm(
    query: str,
    references: Sequence[str],
    retrieved_texts: Sequence[str],
    api_key: str,
    endpoint: str,
    model: str,
) -> float:
    try:
        from datasets import Dataset  # type: ignore[import-not-found]
        from langchain_openai import ChatOpenAI  # type: ignore[import-not-found]
        from ragas import evaluate  # type: ignore[import-not-found]
        from ragas.metrics import context_recall  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError(
            "RAGAS LLM mode requires: ragas, datasets, langchain-openai"
        ) from exc

    llm = ChatOpenAI(
        model=model,
        base_url=endpoint,
        api_key=api_key,
        temperature=0,
        default_headers={
            "HTTP-Referer": "https://local.finder.bm25",
            "X-Title": "FINDER BM25 Context Recall",
        },
    )

    data = {
        "question": [query],
        "contexts": [list(retrieved_texts)],
        "ground_truth": ["\n".join(str(r) for r in references)],
        "answer": [""],
    }

    ds = Dataset.from_dict(data)
    result = evaluate(dataset=ds, metrics=[context_recall], llm=llm)
    val = float(result["context_recall"][0])
    return max(0.0, min(100.0, val * 100.0))


def context_recall_ragas_nonllm(
    references,
    retrieved_texts,
) -> float:

    try:
        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics import NonLLMContextRecall
        import asyncio
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc

    sample = SingleTurnSample(
        retrieved_contexts=list(retrieved_texts),
        reference_contexts=list(references),
    )

    metric = NonLLMContextRecall()

    val = asyncio.run(metric.single_turn_ascore(sample))

    return max(0.0, min(100.0, float(val) * 100.0))