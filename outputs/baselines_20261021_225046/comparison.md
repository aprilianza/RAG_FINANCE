# Baseline comparison: baselines_20261021_225046

- Evaluator: ragas_llm (judge: deepseek/deepseek-v4.1-flash)
- Company filter: off (full corpus)
- Corpus: corpus_paragraphs_v3.jsonl (metadata header on)
- Sample: 10.0% (seed 42), top-k 10
- Judge scored: MPR 570/570

## Context Recall (ragas_llm (judge: deepseek/deepseek-v4.1-flash))

| Category | MPR |
|---|---:|
| Accounting | 31.38 |
| Company Overview | 42.23 |
| Financials | 29.54 |
| Footnotes | 32.37 |
| Governance | 23.94 |
| Legal | 36.88 |
| Risk | 54.96 |
| Shareholder Return | 72.09 |
| Total | 38.34 |

## Context Recall (heuristic, deterministic)

| Category | MPR |
|---|---:|
| Accounting | 46.94 |
| Company Overview | 60.65 |
| Financials | 22.22 |
| Footnotes | 34.21 |
| Governance | 34.72 |
| Legal | 51.02 |
| Risk | 85.71 |
| Shareholder Return | 70.41 |
| Total | 47.28 |
