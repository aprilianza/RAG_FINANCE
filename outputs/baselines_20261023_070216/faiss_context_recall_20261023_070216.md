- Evaluator: ragas_llm (judge: deepseek/deepseek-v4.1-flash)
- Company filter: off (full corpus)
- Corpus: corpus_paragraphs_v3.jsonl (metadata header on)
- Queries: 570 (sample 10.0%, seed 42), top-k 10
- Judge scored: 570/570

| Category | FAISS (Context Recall) | FAISS (Heuristic) |
|---|---:|---:|
| Accounting | 19.61 | 36.73 |
| Company Overview | 29.59 | 46.76 |
| Financials | 15.37 | 18.18 |
| Footnotes | 14.22 | 28.95 |
| Governance | 9.46 | 20.14 |
| Legal | 27.01 | 44.90 |
| Risk | 31.37 | 83.67 |
| Shareholder Return | 41.33 | 58.16 |
| Total | 22.10 | 38.60 |
