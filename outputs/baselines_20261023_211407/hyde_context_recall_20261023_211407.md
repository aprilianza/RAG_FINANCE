- Evaluator: ragas_llm (judge: deepseek/deepseek-v4.1-flash)
- Company filter: off (full corpus)
- Corpus: corpus_paragraphs_v3.jsonl (metadata header on)
- Queries: 570 (sample 10.0%, seed 42), top-k 10
- Judge scored: 569/570
- WARNING: 1 queries have no judge score; Context Recall above averages only the scored ones. Fill them in with rescore_judge.py.

| Category | HyDE (Context Recall) | HyDE (Heuristic) |
|---|---:|---:|
| Accounting | 24.87 | 38.78 |
| Company Overview | 34.04 | 55.09 |
| Financials | 17.18 | 16.67 |
| Footnotes | 25.10 | 33.16 |
| Governance | 13.85 | 22.92 |
| Legal | 21.58 | 36.73 |
| Risk | 31.11 | 77.55 |
| Shareholder Return | 51.43 | 63.27 |
| Total | 26.45 | 40.35 |
