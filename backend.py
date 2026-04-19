import math
import ast
import re
from collections import defaultdict
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from FrontEnd_Support_main import compute_tfidf, load_index, load_stopwords, preprocess, process_query


app = FastAPI(title="VSM Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    alpha: float = 0.005


class SearchResult(BaseModel):
    doc_id: str
    score: float
    title: str


class GraphNode(BaseModel):
    id: str
    label: str
    size: float
    type: str


class GraphLink(BaseModel):
    source: str
    target: str
    strength: float


class SearchResponse(BaseModel):
    results: List[SearchResult]
    nodes: List[GraphNode]
    links: List[GraphLink]
    query_vector_norm: float
    metrics: Dict[str, float]


stopwords = None
doc_ids = None
tf_index = None
df_index = None
doc_vectors = None
query_relevance = None


def canonical_query_key(query_text: str) -> str:
    # Canonicalize for robust lookup against the relevance file.
    normalized = " ".join(query_text.strip().lower().split())
    if not normalized:
        return ""
    return " ".join(preprocess(normalized, stopwords or set()))


def parse_query_relevance(path: str) -> Dict[str, set[str]]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    query_pattern = re.compile(r"^[Qq]uery\s*[=:]\s*['\"]?(.*?)['\"]?\s*$")
    set_pattern = re.compile(r"^\{.*\}$")

    relevance: Dict[str, set[str]] = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        match = query_pattern.match(line)
        if match:
            query_text = match.group(1).strip().strip("'\"")
            expected_docs: set[str] = set()

            j = i + 1
            while j < len(lines):
                candidate = lines[j].strip()
                if set_pattern.match(candidate):
                    try:
                        raw = ast.literal_eval(candidate)
                        expected_docs = {str(value) for value in raw}
                    except (ValueError, SyntaxError):
                        expected_docs = set()
                    i = j
                    break
                j += 1

            relevance[canonical_query_key(query_text)] = expected_docs
        i += 1

    return relevance


def compute_metrics(retrieved: set[str], expected: set[str], total_docs: int) -> Dict[str, float]:
    tp = len(retrieved & expected)
    fp = len(retrieved - expected)
    fn = len(expected - retrieved)
    tn = total_docs - tp - fp - fn

    precision = tp / len(retrieved) if retrieved else 0.0
    recall = tp / len(expected) if expected else 0.0
    accuracy = (tp + tn) / total_docs if total_docs else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1": f1,
    }


def load_speech_preview(doc_id: str, max_chars: int = 100) -> str:
    try:
        with open(f"Speeches/speech_{doc_id}.txt", "r", encoding="utf-8", errors="ignore") as f:
            return f.read(max_chars).replace("\n", " ")[:max_chars]
    except OSError:
        return ""


def load_speech_text(doc_id: str) -> str:
    try:
        with open(f"Speeches/speech_{doc_id}.txt", "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except OSError:
        return ""


@app.on_event("startup")
async def startup_event() -> None:
    global stopwords, doc_ids, tf_index, df_index, doc_vectors, query_relevance
    stopwords = load_stopwords("stopwords.txt")
    doc_ids, tf_index, df_index = load_index("index")
    doc_vectors = compute_tfidf(tf_index, df_index, doc_ids)
    query_relevance = parse_query_relevance("queries.txt")


@app.get("/api/health")
async def health() -> Dict[str, int | str]:
    return {"status": "ok", "documents": len(doc_ids) if doc_ids else 0}


@app.get("/api/stats")
async def get_stats() -> Dict[str, float | int]:
    if not doc_ids:
        raise HTTPException(status_code=500, detail="Indexes not loaded")

    return {
        "total_documents": len(doc_ids),
        "vocabulary_size": len(df_index),
        "avg_df": sum(df_index.values()) / len(df_index) if df_index else 0,
    }


@app.get("/api/speeches/{doc_id}")
async def get_speech(doc_id: str) -> Dict[str, str]:
    if not doc_ids:
        raise HTTPException(status_code=500, detail="Indexes not loaded")

    if doc_id not in doc_ids:
        raise HTTPException(status_code=404, detail="Speech not found")

    text = load_speech_text(doc_id)
    if not text:
        raise HTTPException(status_code=404, detail="Speech file is missing")

    return {"doc_id": doc_id, "text": text}


@app.post("/api/search", response_model=SearchResponse)
async def search(req: QueryRequest) -> SearchResponse:
    if not doc_vectors:
        raise HTTPException(status_code=500, detail="Indexes not loaded")

    ranked = process_query(req.query, stopwords, df_index, doc_vectors, doc_ids, alpha=req.alpha)
    retrieved_all = {doc_id for doc_id, _score in ranked}
    expected = (query_relevance or {}).get(canonical_query_key(req.query), set())
    metrics = compute_metrics(retrieved_all, expected, len(doc_ids))

    if not ranked:
        return SearchResponse(results=[], nodes=[], links=[], query_vector_norm=0.0, metrics=metrics)

    results = [
        SearchResult(doc_id=doc_id, score=score, title=load_speech_preview(doc_id, 80))
        for doc_id, score in ranked[:20]
    ]

    top_doc_ids = [doc_id for doc_id, _ in ranked[:10]]
    nodes = [GraphNode(id="query", label="Query", size=15, type="query")]
    links = []

    for doc_id in top_doc_ids:
        score = next((s for d, s in ranked if d == doc_id), 0.0)
        nodes.append(
            GraphNode(
                id=f"doc_{doc_id}",
                label=f"Doc {doc_id}",
                size=max(5, score * 30),
                type="document",
            )
        )
        links.append(GraphLink(source="query", target=f"doc_{doc_id}", strength=score))

    tokens = preprocess(req.query, stopwords)
    q_tf = defaultdict(int)
    for token in tokens:
        q_tf[token] += 1

    n_docs = len(doc_ids)
    query_vec = {}
    for term, tf in q_tf.items():
        df = df_index.get(term, 0)
        if df == 0:
            continue
        idf = math.log10(n_docs / df)
        query_vec[term] = (1 + math.log10(tf)) * idf if tf > 0 else 0.0

    query_norm = math.sqrt(sum(value * value for value in query_vec.values()))

    return SearchResponse(
        results=results,
        nodes=nodes,
        links=links,
        query_vector_norm=query_norm,
        metrics=metrics,
    )