# Vector Space Model (VSM) Search Engine

This project is an Information Retrieval assignment that implements a **Vector Space Model (VSM)** using:

- TF-IDF weighting
- Cosine similarity ranking
- Configurable similarity threshold `alpha`
- Query evaluation against expected relevance sets

It includes:

- A Python backend API (FastAPI)
- A React + Vite frontend UI
- CLI scripts for indexing/searching/evaluation

## Project Structure

```text
Assignment 2/
|- backend.py                  # FastAPI backend
|- FrontEnd_Support_main.py    # Core IR logic (preprocess, index, tf-idf, query scoring)
|- main.py                     # CLI search interface
|- evaluate.py                 # Query-set evaluation script
|- requirements.txt            # Python dependencies
|- queries.txt                 # Evaluation queries + expected document sets
|- stopwords.txt               # Stopword list
|- index/                      # Saved inverted index files
|- Speeches/                   # Corpus documents
`- frontend/                   # React frontend (Vite + Tailwind)
```

## Features

- Text preprocessing with `textacy` + tokenization/stemming using `nltk`
- Stopword removal (stemmed stopword matching)
- Inverted index persistence:
  - `index/doc_ids.txt`
  - `index/df_index.txt`
  - `index/tf_index.txt`
- Ranked search results with cosine similarity
- Per-query metrics from backend:
  - Precision
  - Recall
  - Accuracy
  - F1
- Frontend can expand a result to load full speech text

## Prerequisites

- Python 3.10+
- Node.js 18+
- npm

## 1) Python Setup

From project root:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Download required NLTK tokenizer data (first run only):

```powershell
python -c "import nltk; nltk.download('punkt')"
```

## 2) Run Backend API

From project root:

```powershell
uvicorn backend:app --host 127.0.0.1 --port 8001 --reload
```

Backend base URL:

`http://127.0.0.1:8001`

## 3) Run Frontend

Open a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Frontend URL:

`http://localhost:5173`

The Vite dev server proxies `/api` requests to `http://localhost:8001`.

## API Endpoints

- `GET /api/health`
  - Returns backend status and indexed document count.
- `GET /api/stats`
  - Returns corpus stats (`total_documents`, `vocabulary_size`, `avg_df`).
- `POST /api/search`
  - Body:
    ```json
    {
      "query": "Hillary Clinton",
      "alpha": 0.005
    }
    ```
  - Returns ranked results, graph nodes/links, query norm, and metrics.
- `GET /api/speeches/{doc_id}`
  - Returns full speech text for a document ID.

## CLI Usage

### Interactive Search

```powershell
python main.py
```

If `index/` exists, it loads saved indexes. Otherwise it builds index files from `Speeches/`.

### Evaluation Script

```powershell
python evaluate.py
```

Optional arguments:

```powershell
python evaluate.py queries.txt --alpha 0.005 --speeches_dir Speeches --index_dir index --stopwords stopwords.txt
```

This prints per-query and average metrics (Precision, Recall, F1).

## Notes

- Keep `Speeches/`, `stopwords.txt`, and `queries.txt` in the project root as expected by the scripts.
- The current frontend only depends on backend `/api` routes; run backend first for full functionality.

## Tech Stack

- Backend: FastAPI, Pydantic, Uvicorn
- IR/NLP: textacy, nltk
- Frontend: React, Vite, Tailwind CSS, Axios
