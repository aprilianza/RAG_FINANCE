# Baseline comparison: baselines_20261018_155709

- Evaluator: ragas_llm (judge: deepseek/deepseek-v4.1-flash)
- Company filter: off (full corpus)
- Corpus: corpus_paragraphs_v2.jsonl (metadata header off)
- Sample: 10.0% (seed 42), top-k 10
- Judge scored: MPR 570/570

## Context Recall (ragas_llm (judge: deepseek/deepseek-v4.1-flash))

| Category | MPR |
|---|---:|
| Accounting | 14.59 |
| Company Overview | 18.74 |
| Financials | 10.30 |
| Footnotes | 13.65 |
| Governance | 8.58 |
| Legal | 22.01 |
| Risk | 13.62 |
| Shareholder Return | 19.34 |
| Total | 14.68 |

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
