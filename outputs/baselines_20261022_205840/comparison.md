# Baseline comparison: baselines_20261022_205840

- Evaluator: ragas_llm (judge: deepseek/deepseek-v4.1-flash)
- Company filter: off (full corpus)
- Corpus: corpus_paragraphs_v3.jsonl (metadata header on)
- Sample: 10.0% (seed 42), top-k 10
- Judge scored: BM25 569/570
- WARNING: some queries have no judge score; their method averages cover only scored queries. Fill them in with rescore_judge.py.

## Context Recall (ragas_llm (judge: deepseek/deepseek-v4.1-flash))

| Category | BM25 |
|---|---:|
| Accounting | 17.77 |
| Company Overview | 17.34 |
| Financials | 10.61 |
| Footnotes | 19.05 |
| Governance | 7.81 |
| Legal | 20.16 |
| Risk | 15.13 |
| Shareholder Return | 24.70 |
| Total | 15.98 |

## Context Recall (heuristic, deterministic)

| Category | BM25 |
|---|---:|
| Accounting | 42.86 |
| Company Overview | 34.26 |
| Financials | 9.09 |
| Footnotes | 32.63 |
| Governance | 19.44 |
| Legal | 28.57 |
| Risk | 63.27 |
| Shareholder Return | 36.73 |
| Total | 30.70 |
