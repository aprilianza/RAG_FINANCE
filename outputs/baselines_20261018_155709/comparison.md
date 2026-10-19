# Baseline comparison: baselines_20261018_155709

- Evaluator: ragas_llm (judge: deepseek/deepseek-v4.1-flash)
- Company filter: off (full corpus)
- Corpus: corpus_paragraphs_v2.jsonl
- Sample: 10.0% (seed 42), top-k 10

## Context Recall (ragas_llm (judge: deepseek/deepseek-v4.1-flash))

| Category | MPR |
|---|---:|
| Accounting | 39.74 |
| Company Overview | 50.39 |
| Financials | 41.17 |
| Footnotes | 49.20 |
| Governance | 53.91 |
| Legal | 52.06 |
| Risk | 50.24 |
| Shareholder Return | 49.46 |
| Total | 48.17 |

## Context Recall (heuristic, deterministic)

| Category | MPR |
|---|---:|
| Accounting | 16.33 |
| Company Overview | 28.70 |
| Financials | 5.56 |
| Footnotes | 11.58 |
| Governance | 17.36 |
| Legal | 18.37 |
| Risk | 38.78 |
| Shareholder Return | 18.37 |
| Total | 18.42 |
