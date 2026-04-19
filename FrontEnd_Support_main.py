import os
import math
import re
from collections import defaultdict
from pathlib import Path
import re
import textacy.preprocessing as pp #type: ignore
from nltk.tokenize import word_tokenize  #type: ignore
from nltk.stem import   PorterStemmer #type: ignore
from nltk.stem import SnowballStemmer #type: ignore

stemmer = SnowballStemmer("english")
temp_tfid = -1
phrasequery = None
complexquery = None
ui_card = None

# ── 1. LOAD STOP WORDS ────────────────────────────────────────────────────────
def load_stopwords(path):
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8", errors="ignore") as f:
        raw = set(line.strip().lower() for line in f if line.strip())
    return set(stemmer.stem(w) for w in raw)

def PreProcessing(content):
    content = pp.normalize.unicode(content) # makes the text encoding normalized
    content = pp.remove.brackets(content)   # removes any brackets and their content from the text
    content = pp.normalize.hyphenated_words(content)    # normalizes hyphenated words by removing the hyphen (e.g., "well-known" becomes "wellknown")
    content = pp.normalize.whitespace(content)  # replaces multiple consecutive whitespace characters with a single space

    content = re.sub(r"[^\w\s]", " ", content)  # removes any punctuation by replacing non-word and non-space characters with a space

    return content

# ── 2. PREPROCESSING PIPELINE ────────────────────────────────────────────────
def preprocess(text, stopwords):
    # case folding

    # do preprocessing on the file content
    text = PreProcessing(text)
    
    # Break sentences into tokens
    tokens = word_tokenize(text.lower())
    
    tokens = [t for t in tokens if t not in stopwords]
    
    # use stemmization to reduce the token to its root form using ntlk library stemmer
    stemmed = [stemmer.stem(token) for token in tokens]
    
    #check if token is alphanumeric to remove any leftover punchuation or special characters
    stemmed = [t for t in stemmed if t.isalnum()]

    # stop-word removal
    
    return stemmed


# ── 3. BUILD TF AND DF INDEXES ────────────────────────────────────────────────
def build_index(speeches_dir, stopwords):
    """
    Returns:
        doc_ids   : list of document identifiers (filenames)
        tf_index  : { term: { doc_id: raw_tf } }
        df_index  : { term: df_count }
    """
    tf_index = defaultdict(lambda: defaultdict(int))
    df_index = defaultdict(int)
    doc_ids = []

    files = sorted(
        [f for f in os.listdir(speeches_dir) if f.endswith(".txt")]
    )

    for fname in files:
        doc_id = os.path.splitext(fname)[0].split("_")[1]  # e.g., "12345_Speech.txt" -> "12345"
        doc_ids.append(doc_id)
        path = os.path.join(speeches_dir, fname)
        with open(path, encoding="utf-8", errors="ignore") as f:
            text = f.read()
        tokens = preprocess(text, stopwords)
        seen = set()
        for token in tokens:
            tf_index[token][doc_id] += 1
            if token not in seen:
                df_index[token] += 1
                seen.add(token)

    return doc_ids, tf_index, df_index


# ── 4. COMPUTE TF-IDF WEIGHTS ─────────────────────────────────────────────────
def compute_tfidf(tf_index, df_index, doc_ids):
    """
    IDF = log(N / df)   (log base 10)
    TF-IDF weight = tf * idf
    Returns: { doc_id: { term: weight } }
    """
    N = len(doc_ids)
    doc_vectors = defaultdict(dict)

    for term, doc_tf in tf_index.items():
        df = df_index[term]
        idf = math.log10(N / df) if df > 0 else 0
        for doc_id, tf in doc_tf.items():
            # Better (log-normalized tf):
            doc_vectors[doc_id][term] = (1 + math.log10(tf)) * idf

    return doc_vectors


# ── 5. COSINE SIMILARITY ──────────────────────────────────────────────────────
def cosine_similarity(query_vec, doc_vec):
    # dot product
    dot = sum(query_vec.get(t, 0) * doc_vec.get(t, 0) for t in query_vec)
    # magnitudes
    mag_q = math.sqrt(sum(v * v for v in query_vec.values()))
    mag_d = math.sqrt(sum(v * v for v in doc_vec.values()))
    if mag_q == 0 or mag_d == 0:
        return 0.0
    return dot / (mag_q * mag_d)


# ── 6. QUERY PROCESSING ───────────────────────────────────────────────────────
def process_query(query_text, stopwords, df_index, doc_vectors, doc_ids,
                  alpha=0.005):
    """
    Tokenise query, compute tf-idf query vector, score all docs,
    filter by alpha threshold, return sorted list of (doc_id, score).
    """
    N = len(doc_ids)
    tokens = preprocess(query_text, stopwords)

    # build query tf
    q_tf = defaultdict(int)
    for t in tokens:
        q_tf[t] += 1

    # build query tfidf vector
    query_vec = {}
    for term, tf in q_tf.items():
        df = df_index.get(term, 0)
        if df == 0:
            continue
        idf = math.log10(N / df)
        query_vec[term] = (1 + math.log10(tf)) * idf if tf > 0 else 0

    if not query_vec:
        return []

    results = []
    for doc_id, doc_vec in doc_vectors.items():
        score = cosine_similarity(query_vec, doc_vec)
        if score >= alpha:
            results.append((doc_id, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ── 7. SAVE / LOAD INDEXES ────────────────────────────────────────────────────
def save_index(tf_index, df_index, doc_ids, out_dir="index"):
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "doc_ids.txt"), "w") as f:
        for d in doc_ids:
            f.write(d + "\n")

    with open(os.path.join(out_dir, "df_index.txt"), "w") as f:
        for term, df in df_index.items():
            f.write(f"{term}\t{df}\n")

    with open(os.path.join(out_dir, "tf_index.txt"), "w") as f:
        for term, doc_tf in tf_index.items():
            postings = ",".join(f"{d}:{c}" for d, c in doc_tf.items())
            f.write(f"{term}\t{postings}\n")

    print(f"[INFO] Index saved to '{out_dir}/'")


def load_index(out_dir="index"):
    doc_ids = []
    with open(os.path.join(out_dir, "doc_ids.txt")) as f:
        doc_ids = [line.strip() for line in f]

    df_index = {}
    with open(os.path.join(out_dir, "df_index.txt")) as f:
        for line in f:
            term, df = line.strip().split("\t")
            df_index[term] = int(df)

    tf_index = defaultdict(lambda: defaultdict(int))
    with open(os.path.join(out_dir, "tf_index.txt")) as f:
        for line in f:
            parts = line.strip().split("\t")
            term = parts[0]
            if len(parts) > 1:
                for entry in parts[1].split(","):
                    d, c = entry.rsplit(":", 1)
                    tf_index[term][d] = int(c)

    print(f"[INFO] Index loaded from '{out_dir}/'")
    return doc_ids, tf_index, df_index


# ── 8. COMMAND-LINE INTERFACE ─────────────────────────────────────────────────

if __name__ == "__main__":
    stopwords = load_stopwords("stopwords.txt")
    SPEECHES_DIR = "Speeches"
    
    # Build or load index
    index_dir = "index"
    if os.path.exists(index_dir):
        choice = input("Saved index found. Load it? (y/n): ").strip().lower()
        if choice == "y":
            doc_ids, tf_index, df_index = load_index(index_dir)
        else:
            print("[INFO] Building index …")
            doc_ids, tf_index, df_index = build_index(SPEECHES_DIR, stopwords)
            save_index(tf_index, df_index, doc_ids, index_dir)
    else:
        print("[INFO] Building index …")
        doc_ids, tf_index, df_index = build_index(SPEECHES_DIR, stopwords)
        save_index(tf_index, df_index, doc_ids, index_dir)

    print(f"[INFO] {len(doc_ids)} documents indexed, {len(df_index)} unique terms.")

    # Compute TF-IDF doc vectors
    doc_vectors = compute_tfidf(tf_index, df_index, doc_ids)

    alpha = 0.005
    print(f"\nVSM ready  (alpha threshold = {alpha})")
    print("Type a query and press Enter. Type 'quit' to exit.\n")

    while True:
        query = input("Query> ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        results = process_query(query, stopwords, df_index, doc_vectors,
                                doc_ids, alpha=alpha)
        if not results:
            print("  No documents above threshold.\n")
        else:
            print("  Results:")
            print(f"{[doc_id for doc_id, _ in results]} ")
