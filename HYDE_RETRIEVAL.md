# HyDE Retrieval Setup & Usage

## Overview
HyDE (Hypothetical Document Embeddings) is a retrieval method that generates hypothetical documents from a query using an LLM, then finds similar documents in the corpus based on embedding similarity.

## Installation

Install dependencies including HyDE support:
```bash
pip install -r requirements.txt
```

### Key Dependencies
- `transformers>=4.36.0` - For Qwen2.5-1.5B model
- `sentence-transformers>=2.2.0` - For embedding model
- `langchain>=0.1.0` - For LLM orchestration
- `torch>=2.0.0` - For model inference

## Configuration

### Via settings.py (Default)
Edit `finder_bm25/settings.py`:
```python
CONFIG = AppConfig(
    # ... other params ...
    retrieval_method="hyde",  # or "bm25" for BM25
    embedding_model="all-MiniLM-L6-v2",
    hyde_llm_model="Qwen/Qwen2.5-1.5B",
    hyde_device="cpu",  # or "cuda" for GPU
    hyde_num_hypothetical_docs=3,
    # ... rest of config ...
)
```

### Via CLI Arguments
```powershell
python run_finder_bm25.py `
  --retrieval-method hyde `
  --embedding-model all-MiniLM-L6-v2 `
  --hyde-llm-model Qwen/Qwen2.5-1.5B `
  --hyde-device cpu `
  --hyde-num-hypothetical-docs 3 `
  --experiment-name hyde_exp1
```

## Running HyDE Retrieval

### Basic Usage (10% sample)
```powershell
python run_finder_bm25.py --retrieval-method hyde --experiment-name hyde_test
```

### Full Dataset with GPU
```powershell
python run_finder_bm25.py `
  --retrieval-method hyde `
  --hyde-device cuda `
  --sample-percent 100.0 `
  --experiment-name hyde_full_gpu
```

### Compare BM25 vs HyDE (same experiment folder)
```powershell
# Run BM25
python run_finder_bm25.py --retrieval-method bm25 --experiment-name comparison

# Run HyDE
python run_finder_bm25.py --retrieval-method hyde --experiment-name comparison
```

## How HyDE Works

1. **Generate Hypothetical Documents**: Uses Qwen2.5-1.5B to generate `hyde_num_hypothetical_docs` (default 3) documents that would answer the query.
2. **Embed Hypothetical Docs**: Encodes generated documents using the embedding model.
3. **Compute Mean Embedding**: Averages the hypothetical document embeddings.
4. **Similarity Search**: Finds top-k paragraphs most similar to the mean hypothetical embedding using cosine similarity.
5. **Return Results**: Returns top-k paragraphs with similarity scores.

## Performance Notes

### CPU vs GPU
- **CPU**: Slower, suitable for development/testing. Default is CPU for portability.
- **GPU**: Faster, recommended for large-scale experiments. Set `hyde_device="cuda"`.

### Model Selection
- **Embedding Model**: `all-MiniLM-L6-v2` is lightweight (22M params), `all-mpnet-base-v2` is larger but higher quality.
- **LLM Model**: `Qwen/Qwen2.5-1.5B` is small; alternative: `Qwen/Qwen2-7B` (larger, slower) or `meta-llama/Llama-2-7b` (not allowed by Meta's license for commercial use on some platforms).

### Corpus Cache
HyDE caches corpus embeddings (similar to BM25 tokenized corpus) to speed up subsequent runs:
```
outputs/
  ├── corpus_paragraphs.jsonl  (paragraphs, shared with BM25)
  └── hyde_embeddings.npy      (corpus embeddings, if you add this feature)
```

## Troubleshooting

### OOM Errors
- Reduce `top_k` or `sample_percent`
- Use CPU instead of GPU
- Use smaller embedding model

### LLM Too Slow
- Use smaller LLM model (e.g., `distilgpt2` instead of Qwen)
- Reduce `hyde_num_hypothetical_docs`
- Use GPU

### CUDA Not Available
Ensure torch is installed with CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Comparing Results

Results are saved in:
- `outputs/experiment_name/bm25_context_recall_TIMESTAMP.md` (metrics table)
- `outputs/experiment_name/bm25_context_recall_TIMESTAMP.csv` (metrics CSV)
- `outputs/experiment_name/bm25_context_recall_TIMESTAMP_config.json` (run config)
- `outputs/experiment_name/bm25_context_recall_TIMESTAMP_retrieval_details.jsonl` (per-query details)

Compare metrics across multiple runs by loading the CSV files and plotting.
