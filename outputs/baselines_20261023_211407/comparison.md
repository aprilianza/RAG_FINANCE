# Baseline comparison: baselines_20261023_211407

- Evaluator: ragas_llm (judge: deepseek/deepseek-v4.1-flash)
- Company filter: off (full corpus)
- Corpus: corpus_paragraphs_v3.jsonl (metadata header on)
- Sample: 10.0% (seed 42), top-k 10
- Judge scored: HyDE 569/570
- WARNING: some queries have no judge score; their method averages cover only scored queries. Fill them in with rescore_judge.py.

## Context Recall (ragas_llm (judge: deepseek/deepseek-v4.1-flash))

| Category | HyDE |
|---|---:|
| Accounting | 24.87 |
| Company Overview | 34.04 |
| Financials | 17.18 |
| Footnotes | 25.10 |
| Governance | 13.85 |
| Legal | 21.58 |
| Risk | 31.11 |
| Shareholder Return | 51.43 |
| Total | 26.45 |

## Context Recall (heuristic, deterministic)

| Category | HyDE |
|---|---:|
| Accounting | 38.78 |
| Company Overview | 55.09 |
| Financials | 16.67 |
| Footnotes | 33.16 |
| Governance | 22.92 |
| Legal | 36.73 |
| Risk | 77.55 |
| Shareholder Return | 63.27 |
| Total | 40.35 |
