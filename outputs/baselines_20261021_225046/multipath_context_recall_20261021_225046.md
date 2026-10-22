- Evaluator: ragas_llm (judge: deepseek/deepseek-v4.1-flash)
- Company filter: off (full corpus)
- Corpus: corpus_paragraphs_v3.jsonl (metadata header on)
- Queries: 570 (sample 10.0%, seed 42), top-k 10
- Judge scored: 570/570

| Category | MPR (Context Recall) | MPR (Heuristic) |
|---|---:|---:|
| Accounting | 31.38 | 46.94 |
| Company Overview | 42.23 | 60.65 |
| Financials | 29.54 | 22.22 |
| Footnotes | 32.37 | 34.21 |
| Governance | 23.94 | 34.72 |
| Legal | 36.88 | 51.02 |
| Risk | 54.96 | 85.71 |
| Shareholder Return | 72.09 | 70.41 |
| Total | 38.34 | 47.28 |
