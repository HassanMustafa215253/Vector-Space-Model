import os
import math
import re
from collections import defaultdict
import textacy.preprocessing as pp
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

"""Note: I have purposely commented the code, and are not leftovers of AI"""


speech_dir = "Speeches"
index_dir    = "index"
stop_path = "stopwords.txt"
ALPHA = 0.005

stemmer = SnowballStemmer("english")

def Preprocessing(content):
    content = pp.normalize.unicode(content)
    content = pp.remove.brackets(content)
    content = pp.normalize.hyphenated_words(content)
    content = pp.normalize.whitespace(content)
    content = re.sub(r"[^\w\s]", " ", content)

    return content


#Load stopwords
stopwords = set()
if os.path.exists(stop_path):
    with open(stop_path, encoding="utf-8", errors="ignore") as f:
        raw_stopwords = set(line.strip().lower() for line in f if line.strip())
    # we are stemming stopwords here so we don't have to stem them every time during query processing
    stopwords = set(stemmer.stem(w) for w in raw_stopwords)


tf_index = defaultdict(lambda: defaultdict(int))
df_index = defaultdict(int)
doc_ids  = []

if os.path.exists(index_dir):
    
    #load doc_ids
    with open(os.path.join(index_dir, "doc_ids.txt")) as f:
        doc_ids = [line.strip() for line in f]

    #load df_index
    with open(os.path.join(index_dir, "df_index.txt")) as f:
        for line in f:
            term, df = line.strip().split("\t")
            df_index[term] = int(df)

    #load tf_index
    with open(os.path.join(index_dir, "tf_index.txt")) as f:
        for line in f:
            parts = line.strip().split("\t")
            term = parts[0]
            if len(parts) > 1:
                for entry in parts[1].split(","):
                    d, c = entry.rsplit(":", 1)
                    tf_index[term][d] = int(c)

else:

    files = sorted([f for f in os.listdir(speech_dir) if f.endswith(".txt")])

    for fname in files:
        doc_id = os.path.splitext(fname)[0].split("_")[1]
        doc_ids.append(doc_id)

        with open(os.path.join(speech_dir, fname), encoding="utf-8", errors="ignore") as f:
            text = f.read()

        text = Preprocessing(text)

        tokens  = word_tokenize(text.lower())
        tokens  = [t for t in tokens if t not in stopwords]
        stemmed = [stemmer.stem(t) for t in tokens]
        stemmed = [t for t in stemmed if t.isalnum()]

        seen = set()
        for token in stemmed:
            tf_index[token][doc_id] += 1
            if token not in seen:
                df_index[token] += 1
                seen.add(token)

    #save index
    os.makedirs(index_dir, exist_ok=True)

    with open(os.path.join(index_dir, "doc_ids.txt"), "w") as f:
        for d in doc_ids:
            f.write(d + "\n")

    with open(os.path.join(index_dir, "df_index.txt"), "w") as f:
        for term, df in df_index.items():
            f.write(f"{term}\t{df}\n")

    with open(os.path.join(index_dir, "tf_index.txt"), "w") as f:
        for term, doc_tf in tf_index.items():
            postings = ",".join(f"{d}:{c}" for d, c in doc_tf.items())
            f.write(f"{term}\t{postings}\n")

print(f"{len(doc_ids)} documents indexed, {len(df_index)} unique terms.")


#Compute TF IDF 
N = len(doc_ids)
doc_vectors = defaultdict(dict)

for term, doc_tf in tf_index.items():
    df  = df_index[term]
    idf = math.log10(N / df) if df > 0 else 0
    for doc_id, tf in doc_tf.items():
        doc_vectors[doc_id][term] = (1 + math.log10(tf)) * idf

# User Interface

print("Type a query and press Enter. Type 'quit' to exit.\n")

while True:
    query = input("Query> ").strip()
    if query.lower() in ("quit", "exit", "q"):
        break
    if not query:
        continue

    #preprocess query (same pipeline as documents)
    query_text = Preprocessing(query)

    #Tokenizing queries
    tokens  = word_tokenize(query_text.lower())
    tokens  = [t for t in tokens if t not in stopwords]
    stemmed = [stemmer.stem(t) for t in tokens]
    stemmed = [t for t in stemmed if t.isalnum()]

    #build query tf
    q_tf = defaultdict(int)
    for t in stemmed:
        q_tf[t] += 1

    #build query tfidf vector
    query_vec = {}
    for term, tf in q_tf.items():
        df = df_index.get(term, 0)
        if df == 0:
            continue
        idf = math.log10(N / df)
        query_vec[term] = (1 + math.log10(tf)) * idf if tf > 0 else 0

    if not query_vec:
        print("No documents above threshold.\n")
        continue

    #score all docs via cosine similarity
    results = []
    for doc_id, doc_vec in doc_vectors.items():
        dot   = sum(query_vec.get(t, 0) * doc_vec.get(t, 0) for t in query_vec)
        mag_q = math.sqrt(sum(v * v for v in query_vec.values()))
        mag_d = math.sqrt(sum(v * v for v in doc_vec.values()))
        score = dot / (mag_q * mag_d) if mag_q > 0 and mag_d > 0 else 0.0

        if score >= ALPHA:
            results.append((doc_id, score))

    results.sort(key=lambda x: x[1], reverse=True)

    #display results
    if not results:
        print("  No documents above threshold.\n")
    else:
        print(f"  {len(results)} document(s) retrieved:\n")
        for rank, (doc_id, score) in enumerate(results, 1):
            print(f"  {rank:3}. {doc_id:<30} score={score:.4f}")
        print()