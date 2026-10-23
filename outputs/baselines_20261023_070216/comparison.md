# Baseline comparison: baselines_20261023_070216

- Evaluator: ragas_llm (judge: deepseek/deepseek-v4.1-flash)
- Company filter: off (full corpus)
- Corpus: corpus_paragraphs_v3.jsonl (metadata header on)
- Sample: 10.0% (seed 42), top-k 10
- Judge scored: FAISS 570/570

## Context Recall (ragas_llm (judge: deepseek/deepseek-v4.1-flash))

| Category | FAISS |
|---|---:|
| Accounting | 19.61 |
| Company Overview | 29.59 |
| Financials | 15.37 |
| Footnotes | 14.22 |
| Governance | 9.46 |
| Legal | 27.01 |
| Risk | 31.37 |
| Shareholder Return | 41.33 |
| Total | 22.10 |

## Context Recall (heuristic, deterministic)

| Category | FAISS |
|---|---:|
| Accounting | 36.73 |
| Company Overview | 46.76 |
| Financials | 18.18 |
| Footnotes | 28.95 |
| Governance | 20.14 |
| Legal | 44.90 |
| Risk | 83.67 |
| Shareholder Return | 58.16 |
| Total | 38.60 |
