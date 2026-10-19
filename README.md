# FINDER BM25 / HyDE / FAISS / Multipath Retrieval Recall

Proyek ini adalah implementasi pipeline untuk melakukan retrieval dan evaluasi *context recall* pada dataset **FINDER** berbasis dokumen 10-K perusahaan S&P 500. Pipeline utama mendukung empat metode retrieval:

- **BM25** sebagai baseline lexical retrieval.
- **HyDE (Hypothetical Document Embeddings)** sebagai retrieval berbasis embedding dan generasi dokumen hipotetis.
- **FAISS dense retrieval** sebagai dense retriever berbasis embedding.
- **Multipath Retrieval (MPR)** menggabungkan BM25, FAISS, dan HyDE dengan deduplication dan chunk bundling.

Hasil evaluasi disimpan per eksperimen dalam folder `outputs/` sehingga setiap run dapat dibandingkan dengan mudah.

## Gambaran Umum

Secara garis besar, alur program adalah:

1. Memuat data train dari `dataset/train/`.
2. Membersihkan dan memetakan kategori query ke kategori standar.
3. Melakukan sampling stratified jika diperlukan.
4. Membangun atau memuat korpus paragraf dari file HTML 10-K di `dataset/10k/`.
5. Menjalankan retrieval top-k untuk setiap query.
6. Mengukur *context recall* dengan mode evaluasi yang dipilih.
7. Menyimpan ringkasan hasil, detail per query, dan konfigurasi run ke folder output eksperimen.

## Fitur Utama

- Mendukung retrieval **BM25**, **HyDE**, **FAISS**, dan **Multipath (MPR)**.
- Fitur **Multi-Path Retrieval (MPR)** menjalankan ketiga metode retriever secara paralel, menggabungkan hasil, menghapus duplikat, dan melakukan chunk bundling berdasarkan similarity.
- Mendukung tiga mode evaluasi *context recall*:
  - `heuristic`
  - `ragas_nonllm`
  - `ragas_llm`
- Menyimpan hasil evaluasi dalam format:
  - Markdown
  - CSV
  - JSONL detail retrieval
  - JSON konfigurasi eksperimen
- Mendukung *stratified sampling* untuk mempercepat eksperimen.
- Mem-cache korpus paragraf hasil parsing 10-K agar run berikutnya lebih cepat.

## Struktur Project

```text
.
├── main.py                       # Launcher kompatibel lama
├── run_finder_bm25.py            # Entrypoint utama pipeline
├── finder_bm25/                  # Paket modular pipeline
│   ├── config.py                 # Definisi AppConfig dan validasi
│   ├── settings.py               # Konfigurasi default global
│   ├── data.py                   # Load data train dan sampling
│   ├── preprocess.py             # Parsing HTML 10-K menjadi paragraf
│   ├── retrieval.py              # BM25, HyDE, FAISS, dan Multipath (RRF) retrieval
│   ├── company.py                # Deteksi perusahaan di query + filter korpus per perusahaan
│   ├── evaluation.py             # Metrik context recall
│   ├── pipeline.py               # Orkestrasi end-to-end
│   └── utils.py                  # Utilitas umum
├── dataset/
│   ├── train/                    # Data train parquet
│   └── 10k/                      # File HTML 10-K
├── outputs/                      # Cache korpus dan hasil eksperimen
├── finder.md                     # Dokumen referensi dataset FINDER
├── HYDE_RETRIEVAL.md             # Catatan penggunaan HyDE
└── requirements.txt              # Dependensi Python
```

## Kebutuhan Sistem

Rekomendasi lingkungan:

- Python 3.10 atau lebih baru.
- `pip` terbaru.
- Ruang disk yang cukup untuk file HTML 10-K dan hasil output.
- GPU opsional untuk HyDE, tetapi tidak wajib.

## Instalasi

1. Masuk ke folder project.
2. Install dependensi:

```powershell
pip install -r requirements.txt
```

Jika Anda ingin menjalankan HyDE secara lokal, pastikan dependensi berikut tersedia dan model yang dipakai dapat diunduh:

- `transformers`
- `sentence-transformers`
- `torch`
- `langchain`
- `langchain-community`

Jika Anda ingin menjalankan FAISS dense retrieval, instal juga paket FAISS yang sesuai dengan environment Anda. Pada banyak setup pip, paket yang dipakai adalah:

```powershell
pip install faiss-cpu
```

## Data yang Dibutuhkan

Pipeline ini mengharapkan dua sumber data utama:

### 1. Data train

Lokasi default:

```text
dataset/train/
```

File yang dibaca adalah file `*.parquet`.

Kolom wajib pada parquet:

- `text` -> query asli.
- `answer` -> jawaban ground truth.
- `references` -> daftar referensi evidence.
- `category` -> kategori query.

Kategori yang dikenali dan dipetakan ke kategori standar:

- Accounting
- Company Overview
- Financials
- Footnotes
- Governance
- Legal
- Risk
- Shareholder Return

### 2. Korpus dokumen 10-K

Lokasi default:

```text
dataset/10k/
```

Folder ini berisi file HTML 10-K perusahaan, misalnya `AAPL.html`, `AMZN.html`, dan seterusnya.

## Cara Kerja Parsing Korpus

Saat pipeline dijalankan, file HTML 10-K akan diproses menjadi paragraf teks:

- Tag seperti `script`, `style`, dan elemen XBRL tersembunyi dibuang.
- Tabel **disimpan** sebagai teks per baris (`label | $8,259 | 52% ; baris berikutnya`). Simbol `$`, `%`, dan `( )` yang terpisah di sel sendiri digabung kembali ke angkanya. Tabel panjang dipecah per baris dengan header diulang di tiap chunk.
- Teks dibersihkan dari spasi berlebih.
- Paragraf terlalu panjang akan dipecah menjadi chunk yang lebih kecil.
- Paragraf duplikat dan noise disaring.
- Hasilnya disimpan ke cache `outputs/corpus_paragraphs_v2.jsonl`.

Sufiks `v2` (`CORPUS_VERSION` di `finder_bm25/config.py`) menandai format korpus. Cache lama `corpus_paragraphs.jsonl` (tanpa tabel) tidak dipakai lagi, sehingga run pertama setelah update akan mem-parsing ulang semua HTML dan menghitung ulang embedding.

## Filter Perusahaan (ablation, default off)

Secara default retrieval dilakukan di **seluruh korpus**, sama seperti baseline di paper FINDER (Tabel 4) dan shared vector store di FinSage. Hanya setting ini yang sebanding dengan angka di kedua paper.

Filter perusahaan adalah setting ablation ("company routing") yang diaktifkan dengan `--company-filter`. Filter ini tidak memakai ground truth, hanya teks query dan korpus. Namun hasilnya tidak sebanding dengan baseline FINDER, dan aturan ticker ambigu disusun sambil melihat query evaluasi, jadi laporkan sebagai ablation terpisah.

Query FINDER hampir selalu menyebut satu perusahaan (ticker seperti `AMAT`, `MCK`, atau nama seperti `Hormel Foods`). Jika filter aktif, retrieval hanya dilakukan pada paragraf 10-K milik perusahaan yang terdeteksi:

- Ticker dicocokkan dengan nama file 10-K (`AMAT.html` → `AMAT`).
- Nama perusahaan diambil dari tag inline XBRL `dei:EntityRegistrantName` dan di-cache ke `outputs/company_names.json`.
- Ticker pendek atau yang mirip singkatan (`A`, `D`, `IT`, `CF`, `ON`, ...) hanya diterima jika muncul dalam pola eksplisit, misalnya `(ON)`, `MCK's`, `F:`, `ticker EA`, atau di akhir query. Bagian dari singkatan seperti `D&A`, `R&D`, `D/E` diabaikan.
- Jika beberapa perusahaan disebut (mis. `LUV vs DAL, UAL`), paragraf semua perusahaan tersebut dipakai.
- Jika tidak ada perusahaan yang terdeteksi, retrieval kembali ke seluruh korpus.

Perusahaan yang terdeteksi dan apakah filter diterapkan dicatat per query di file `_retrieval_details.jsonl` (`detected_companies`, `company_filter_applied`).

Jika file cache ini sudah ada, pipeline akan memuat cache tersebut daripada parsing ulang semua HTML.

## Caching Embeddings Dense (HyDE & FAISS)

Untuk retrieval method **HyDE** dan **FAISS**, embeddings dari seluruh corpus (~494K paragraphs) di-compute dan **otomatis disimpan ke cache** di folder `outputs/`:

- **FAISS embeddings**: `outputs/faiss_embeddings_<model_name>_v2.npy`
- **HyDE embeddings**: `outputs/hyde_embeddings_<model_name>_v2.npy`

Caching dilakukan otomatis:

1. **First run**: Embeddings dihitung dan disimpan ke `.npy` file (sekali saja).
2. **Subsequent runs**: Embeddings dimuat langsung dari cache, menghemat waktu computation secara signifikan.

Cache path dihasilkan berdasarkan nama model embedding dan versi korpus (e.g., `BAAI/bge-base-en-v1.5` menjadi `faiss_embeddings_BAAI_bge_base_en_v1.5_v2.npy`). Jika jumlah embedding di cache tidak sama dengan jumlah paragraf korpus, embedding dihitung ulang otomatis.

Jika ingin regenerate embeddings cache (misal ganti embedding model), cukup hapus file `.npy` yang sesuai di folder `outputs/`.

## Entrypoint Program

Ada dua file yang bisa dipakai untuk menjalankan pipeline:

- `run_finder_bm25.py` -> entrypoint utama yang direkomendasikan.
- `rescore_judge.py` -> mengisi ulang skor judge yang hilang/tidak valid dari `_retrieval_details.jsonl` tanpa mengulang retrieval.
- `run_baselines.py` -> menjalankan BM25, FAISS, HyDE, dan Multipath dengan konfigurasi, korpus, sampel, dan judge yang sama, lalu menulis `comparison.md` dan `comparison.csv`.
- `main.py` -> launcher kompatibel lama yang memanggil `run_finder_bm25.py`.

## Konfigurasi Default

Konfigurasi default berada di `finder_bm25/settings.py`.

Saat ini default yang dipakai adalah:

- `retrieval_method = "bm25"` (pilihan: `bm25`, `hyde`, `faiss`, `multipath`)
- `embedding_model = "BAAI/bge-base-en-v1.5"`
- `hyde_llm_model = "Qwen/Qwen2.5-7B-Instruct"`
- `hyde_device = "cuda"`
- `hyde_num_hypothetical_docs = 3`
- `eval_mode = "ragas_llm"`
- `openrouter_model = "deepseek/deepseek-v4.1-flash"`
- `company_filter = False` (full corpus; `True` hanya untuk ablation)
- `rrf_k = 60`
- `multipath_candidates_per_retriever = 30`
- `sample_percent = 10.0`
- `sample_seed = 42`
- `top_k = 10`
- `bm25_k1 = 1.2`
- `bm25_b = 0.75`
- `fuzzy_threshold = 0.85`

Catatan:

- Jika mesin Anda tidak memiliki GPU, ubah `--hyde-device cpu` atau ubah default di `finder_bm25/settings.py`.
- Jika ingin menjalankan baseline lexical saja, gunakan `--retrieval-method bm25`.

## Menjalankan Program

### 1. Jalankan konfigurasi default

```powershell
python run_finder_bm25.py
```

Perintah ini akan menggunakan konfigurasi dari `finder_bm25/settings.py`.

### 2. Jalankan BM25 baseline

```powershell
python run_finder_bm25.py --retrieval-method bm25 --experiment-name bm25_baseline
```

### 3. Jalankan HyDE

```powershell
python run_finder_bm25.py --retrieval-method hyde --experiment-name hyde_experiment
```

### 4. Jalankan FAISS dense retrieval

```powershell
python run_finder_bm25.py --retrieval-method faiss --experiment-name faiss_experiment
```

### 5. Jalankan dengan sampling lebih kecil

```powershell
python run_finder_bm25.py --sample-percent 5 --experiment-name quick_test
```

### 6. Ubah nilai top-k

```powershell
python run_finder_bm25.py --top-k 20 --experiment-name top20_test
```

## Opsi CLI yang Tersedia

Berikut argumen yang bisa dipakai melalui `run_finder_bm25.py`:

- `--root`
  - Override root workspace.
- `--eval-mode`
  - Pilih mode evaluasi: `heuristic`, `ragas_llm`, atau `ragas_nonllm`.
- `--sample-percent`
  - Persentase data train yang dipakai.
- `--sample-seed`
  - Seed random untuk sampling.
- `--top-k`
  - Jumlah paragraf teratas yang diambil dari retrieval.
- `--k1`
  - Parameter BM25 `k1`.
- `--b`
  - Parameter BM25 `b`.
- `--fuzzy-threshold`
  - Ambang kemiripan fuzzy untuk evaluasi heuristic.
- `--openrouter-endpoint`
  - Endpoint OpenRouter untuk mode `ragas_llm`.
- `--openrouter-model`
  - Nama model OpenRouter untuk mode `ragas_llm`.
- `--openrouter-api-key`
  - API key OpenRouter untuk mode `ragas_llm`.
- `--no-progress`
  - Menonaktifkan progress bar.
- `--experiment-name`
  - Nama folder eksperimen di bawah `outputs/`.
- `--retrieval-method`
  - Pilih `bm25`, `hyde`, `faiss`, atau `multipath`.
- `--embedding-model`
  - Model sentence transformer untuk HyDE.
- `--hyde-llm-model`
  - Model Hugging Face untuk generasi dokumen hipotetis.
- `--hyde-device`
  - Pilih `cpu` atau `cuda`.
- `--hyde-num-hypothetical-docs`
  - Jumlah dokumen hipotetis yang dihasilkan per query.
- `--chunk-expansion-similarity-threshold`
  - Similarity threshold untuk chunk bundling dalam multipath retrieval (default: 0.85).
- `--chunk-expansion-max-neighbors`
  - Maksimal tetangga yang dicek per arah untuk chunk bundling dalam multipath retrieval (default: 5).
- `--rrf-k`
  - Konstanta Reciprocal Rank Fusion untuk multipath retrieval (default: 60).
- `--multipath-candidates`
  - Jumlah kandidat per retriever sebelum fusion dalam multipath retrieval (default: 30).
- `--company-filter` / `--no-company-filter`
  - Aktifkan (ablation) atau nonaktifkan filter perusahaan. Default nonaktif.

## Menjalankan Ulang Semua Baseline

Setelah korpus berubah (misalnya `corpus_paragraphs_v2.jsonl` yang kini berisi tabel), semua metode harus dijalankan ulang di korpus, sampel, dan judge yang sama:

```powershell
python run_baselines.py --experiment-name baselines_v2
```

Semua run disimpan di `outputs/baselines_v2/`, plus `comparison.md` dan `comparison.csv`. Opsi lain dari `run_finder_bm25.py` tetap berlaku (mis. `--sample-percent`, `--openrouter-model`, `--company-filter`). Pilih subset metode dengan `--methods bm25 faiss`. Untuk ablation filter perusahaan, jalankan sekali lagi dengan `--company-filter --experiment-name baselines_v2_company`.

## Judge Gagal / Token Habis

Jika panggilan judge gagal (kuota/token OpenRouter habis, rate limit, output tidak ter-parse), skor query itu disimpan sebagai `null` dengan pesan di `eval_error`. Skornya **tidak** dihitung sebagai 0 atau 100. Setelah 5 kegagalan berturut-turut, judge tidak dipanggil lagi, tetapi retrieval dan skor heuristic tetap berjalan sampai selesai. File `.md` dan `comparison.md` menampilkan `Judge scored: X/N` dan peringatan jika ada skor yang hilang. Rata-rata Context Recall hanya dihitung dari query yang punya skor.

Isi skor yang hilang setelah kuota tersedia lagi, tanpa mengulang retrieval atau HyDE:

```powershell
python rescore_judge.py outputs/baselines_v2 --dry-run   # lihat berapa baris yang akan dinilai
python rescore_judge.py outputs/baselines_v2
```

Skrip ini membuat backup `*_retrieval_details.jsonl.bak` sekali, menyimpan progres tiap 10 baris, berhenti sendiri jika judge gagal 5 kali berturut-turut, lalu menulis ulang `.md`, `.csv`, dan `comparison.md`. Jalankan ulang perintah yang sama untuk melanjutkan.

**Run lama (sebelum perbaikan ini)**: kegagalan judge berubah menjadi skor 100, karena RAGAS mengembalikan NaN dan `min(100.0, nan)` di Python bernilai `100.0`. Untuk run seperti itu, tentukan baris yang dinilai ulang secara manual. Misalnya pada `outputs/baselines_20261018_155709`, semua query mulai posisi 346 bernilai 100 setelah token habis:

```powershell
python rescore_judge.py outputs/baselines_20261018_155709 --from-index 346
```

Opsi `--scores 100` menilai ulang semua baris yang bernilai persis 100, termasuk kemungkinan kegagalan sporadis sebelum posisi tersebut. Run lama dengan skor 100 di semua query (mis. `experiment_20260919_110630`) kemungkinan besar terkena bug yang sama dan tidak valid.

## Mode Evaluasi

Apa pun mode yang dipilih, skor `heuristic` selalu ikut dihitung sebagai pembanding deterministik. Skornya muncul sebagai kolom tambahan di `.md` dan `.csv`, dan sebagai `heuristic_score` per query di `_retrieval_details.jsonl`. Header file `.md` mencatat evaluator (termasuk model judge), status filter perusahaan, versi korpus, dan ukuran sampel.

### 1. `heuristic`

Mode ini melakukan pencocokan antara reference evidence dan teks hasil retrieval dengan kombinasi:

- substring match
- sequence similarity
- Jaccard token overlap

Ini adalah mode paling ringan dan cocok untuk eksperimen cepat.

### 2. `ragas_nonllm`

Mode ini memakai metrik `NonLLMContextRecall` dari `ragas`.

Jika mode ini gagal pada salah satu query, skornya disimpan sebagai `null` (lihat "Judge Gagal / Token Habis").

### 3. `ragas_llm`

**Pakai satu model judge yang sama untuk semua run yang dibandingkan.** Pada konteks retrieval yang identik, skor bisa berubah dari 71.47 (`openai/gpt-oss-20b:free`) ke 8.40 (`deepseek/deepseek-v4.1-flash`). Model judge default ada di `openrouter_model` pada `finder_bm25/settings.py`. API key tidak ditulis ke `_config.json`.

Mode ini memakai evaluasi LLM melalui OpenRouter dan `ragas`.

Mode ini membutuhkan:

- koneksi internet
- API key OpenRouter
- paket `ragas`, `datasets`, dan `langchain-openai`

Jika evaluasi LLM gagal, pipeline mencoba ulang hingga 3 kali; jika tetap gagal, skornya disimpan sebagai `null` dan bisa diisi ulang dengan `rescore_judge.py`.

### 4. `faiss`

Mode ini memakai dense embeddings dan FAISS IndexFlatIP untuk mencari paragraf terdekat berdasarkan cosine similarity.

Mode ini cocok jika Anda ingin dense retrieval yang ringan dan deterministik tanpa proses generasi hipotetis seperti HyDE.

## Perbedaan BM25 dan HyDE

### BM25

BM25 adalah retrieval berbasis kata kunci. Metode ini:

- cepat
- stabil
- cocok sebagai baseline
- tidak butuh model generatif

### HyDE

HyDE membuat dokumen hipotetis dari query, lalu mencari paragraf yang paling mirip secara embedding.

Alurnya:

1. Query diubah menjadi beberapa dokumen hipotetis.
2. Dokumen hipotetis di-embed dengan sentence transformer.
3. Embedding dirata-ratakan.
4. Hasilnya dibandingkan dengan embedding paragraf korpus.
5. Diambil top-k paragraf paling mirip.

HyDE biasanya lebih mahal secara komputasi, tetapi bisa lebih baik untuk query singkat dan ambigu.

### FAISS

FAISS di proyek ini dipakai sebagai dense retriever. Alurnya:

1. Query dan paragraf korpus di-embed dengan sentence transformer.
2. Embedding dinormalisasi.
3. FAISS IndexFlatIP digunakan untuk mencari vektor paling mirip.
4. Karena embedding sudah dinormalisasi, inner product dapat dipakai sebagai cosine similarity.

### Multipath Retrieval (MPR)

MPR menggabungkan kekuatan BM25, FAISS, dan HyDE dengan Reciprocal Rank Fusion (RRF) dan chunk bundling:

1. **Parallelisasi**: Ketiga retriever (BM25, FAISS, HyDE) dijalankan secara paralel untuk setiap query. Masing-masing mengambil `multipath_candidates_per_retriever` kandidat (default 30), dibatasi ke perusahaan target jika filter perusahaan aktif.
2. **Fusion dengan RRF**: Skor akhir tiap paragraf adalah `sum(1 / (rrf_k + rank))` dari setiap retriever yang menemukannya. RRF hanya memakai peringkat, jadi skor BM25 (puluhan) dan cosine (0-1) tidak perlu dinormalisasi. Paragraf yang ditemukan beberapa retriever otomatis naik peringkat dan sekaligus terdeduplikasi.
3. **Chunk Expansion/Bundling** (hanya untuk top-k hasil fusion):
   - Chunk tetangga sebelum dan sesudah di dokumen yang sama digabungkan selama cosine similarity-nya dengan chunk anchor melebihi threshold (default 0.85).
   - Ekspansi berhenti pada tetangga pertama di bawah threshold agar teks tetap kontigu, maksimal 5 tetangga per arah.
   - Similarity dihitung dari embedding korpus yang sudah di-cache (tanpa encode ulang).
   - Chunk yang sudah terserap bundle dengan peringkat lebih tinggi tidak dipakai lagi, sehingga top-k tidak berisi teks yang tumpang tindih.
4. **Hasil Akhir**: Top-k bundle diurutkan berdasarkan skor RRF. Hypothetical documents dari jalur HyDE ikut disimpan di `_retrieval_details.jsonl`.

MPR cocok untuk queries yang kompleks dan ambigu karena menggabungkan perspektif lexical, dense, dan generatif.

## Output yang Dihasilkan

Setiap run akan membuat folder eksperimen di bawah `outputs/`.

Contoh:

```text
outputs/
└── experiment_YYYYMMDD_HHMMSS/
    ├── bm25_context_recall_YYYYMMDD_HHMMSS.md
    ├── bm25_context_recall_YYYYMMDD_HHMMSS.csv
    ├── bm25_context_recall_YYYYMMDD_HHMMSS_retrieval_details.jsonl
    └── bm25_context_recall_YYYYMMDD_HHMMSS_config.json
```

Jika retrieval method yang dipakai adalah HyDE, nama file akan memakai prefix `hyde_context_recall_...`.

Jika retrieval method yang dipakai adalah FAISS, nama file akan memakai prefix `faiss_context_recall_...`.

Jika retrieval method yang dipakai adalah Multipath (MPR), nama file akan memakai prefix `multipath_context_recall_...`.

### Isi file output

#### Markdown

File `.md` berisi tabel ringkasan context recall per kategori dan total.

#### CSV

File `.csv` berisi ringkasan yang sama dalam format tabular untuk analisis lanjutan.

#### Retrieval details JSONL

File `.jsonl` berisi detail per query:

- query index
- query asli
- kategori
- skor recall
- references
- top-k paragraf retrieval
- hypothetical documents, jika memakai HyDE

#### Config JSON

File `.json` menyimpan konfigurasi run yang sudah direalisasikan, termasuk:

- path yang dipakai
- timestamp
- nama eksperimen
- lokasi file output

## Contoh Alur Kerja Eksperimen

### Baseline BM25

```powershell
python run_finder_bm25.py `
  --retrieval-method bm25 `
  --eval-mode heuristic `
  --sample-percent 10 `
  --experiment-name bm25_heuristic_10pct
```

### HyDE dengan GPU

```powershell
python run_finder_bm25.py `
  --retrieval-method hyde `
  --hyde-device cuda `
  --eval-mode heuristic `
  --sample-percent 10 `
  --experiment-name hyde_heuristic_10pct
```

### Evaluasi dengan RAGAS LLM

```powershell
python run_finder_bm25.py `
  --retrieval-method bm25 `
  --eval-mode ragas_llm `
  --openrouter-api-key YOUR_OPENROUTER_KEY `
  --openrouter-model nvidia/nemotron-3-nano-30b-a3b:free `
  --experiment-name bm25_ragas_llm
```

### Multi-Path Retrieval (Multipath)

```powershell
python run_finder_bm25.py `
  --retrieval-method multipath `
  --embedding-model BAAI/bge-base-en-v1.5 `
  --hyde-llm-model Qwen/Qwen2.5-7B-Instruct `
  --hyde-device cuda `
  --eval-mode heuristic `
  --chunk-expansion-similarity-threshold 0.85 `
  --chunk-expansion-max-neighbors 5 `
  --experiment-name multipath_full
```

## Troubleshooting

### 1. `No parquet files found`

Pastikan data train ada di `dataset/train/` dan file-nya berekstensi `.parquet`.

### 2. `No HTML files found`

Pastikan file 10-K HTML ada di `dataset/10k/`.

### 3. Error dependency saat menjalankan HyDE

Pastikan paket berikut terpasang:

- `transformers`
- `sentence-transformers`
- `torch`
- `langchain`

Jika tetap gagal, coba jalankan dengan `--retrieval-method bm25` untuk memastikan pipeline dasar sudah jalan.

### 4. CUDA tidak terdeteksi

Jika GPU tidak tersedia, pakai:

```powershell
python run_finder_bm25.py --retrieval-method hyde --hyde-device cpu
```

Atau ubah default `hyde_device` di `finder_bm25/settings.py`.

### 5. Mode `ragas_llm` gagal

Pastikan:

- API key OpenRouter valid.
- Endpoint OpenRouter benar.
- Paket `ragas`, `datasets`, dan `langchain-openai` tersedia.

Jika tetap gagal setelah 3 kali percobaan, skor query tersebut disimpan sebagai `null` dan run tetap berjalan. Lihat bagian "Judge Gagal / Token Habis" untuk mengisinya kembali dengan `rescore_judge.py`.

### 6. Hasil terlalu lambat

Beberapa cara mempercepat run:

- kecilkan `--sample-percent`
- kecilkan `--top-k`
- gunakan `--retrieval-method bm25`
- nonaktifkan progress bar jika tidak diperlukan
- gunakan GPU untuk HyDE jika tersedia

## Catatan Implementasi

- `finder_bm25/settings.py` adalah sumber konfigurasi default yang dipakai pipeline.
- `finder_bm25/config.py` berisi:
  - Validasi parameter penting seperti `top_k`, `sample_percent`, `retrieval_method`, dan parameter chunk expansion.
  - Property untuk cache paths: `faiss_embeddings_cache_path` dan `hyde_embeddings_cache_path`.
- `finder_bm25/pipeline.py` adalah orkestrator utama end-to-end yang mendukung dispatch ke BM25, HyDE, FAISS, dan Multipath.
  - Pass `cache_path` ke `build_faiss_retriever()` dan `build_hyde_retriever()` untuk otomatis caching.
- `finder_bm25/preprocess.py` membangun cache korpus dari HTML 10-K.
- `finder_bm25/company.py` memuat `CompanyIndex` (deteksi ticker/nama perusahaan di query dan indeks paragraf per perusahaan).
- `finder_bm25/retrieval.py` memuat implementasi:
  - Caching embeddings: `_load_or_compute_embeddings()` (lazy load dari cache atau compute & save).
  - Encoding embeddings: `_encode_normalized_embeddings()` (compute normalized embeddings).
  - BM25 retrieval (`retrieve_topk`, `build_bm25`)
  - HyDE retrieval (`retrieve_topk_hyde`, `build_hyde_retriever`, `build_hyde_llm`)
  - FAISS dense retrieval (`retrieve_topk_faiss`, `build_faiss_retriever`)
  - Multipath Retrieval (`retrieve_multipath`, `_rrf_fuse`, `_bundle_neighbors`)
- `finder_bm25/evaluation.py` memuat logika context recall dalam tiga mode (heuristic, ragas_nonllm, ragas_llm).

## Referensi Tambahan

- `finder.md` berisi konteks paper FINDER.
- `HYDE_RETRIEVAL.md` berisi petunjuk khusus penggunaan HyDE.

## Lisensi dan Data

Proyek ini bergantung pada dataset FINDER dan file 10-K yang disediakan pada workspace. Pastikan penggunaan data mengikuti ketentuan dataset dan sumber aslinya.
