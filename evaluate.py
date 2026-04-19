"""
evaluate_queries.py
-------------------
Parses a query/expected-results file, runs each query through the VSM,
and reports precision, recall, and F1 for every query plus overall averages.

Usage:
    python evaluate_queries.py [queries_file] [--alpha 0.005]

Defaults:
    queries_file = "queries.txt"
    alpha        = 0.005
"""

import os
import re
import ast
import sys
import argparse


from FrontEnd_Support_main import *


def parse_query_file(path: str) -> list[dict]:

    with open(path, "r", encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f]

    # Regex patterns
    query_pattern = re.compile(
        r"""^[Qq]uery\s*[=:]\s*['"]?(.*?)['"]?\s*$""",
        re.VERBOSE,
    )
    set_pattern = re.compile(r"^\{.*\}$")

    entries = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        m = query_pattern.match(line)
        if m:
            query_text = m.group(1).strip().strip("'\"")

            # Scan forward for the expected-set line (skip Length= lines)
            expected_set: set[str] | None = None
            j = i + 1
            while j < len(lines):
                candidate = lines[j].strip()
                if set_pattern.match(candidate):
                    try:
                        raw = ast.literal_eval(candidate)
                        expected_set = {str(x) for x in raw}
                    except Exception:
                        pass
                    i = j  # advance outer pointer past the set line
                    break
                j += 1

            entries.append({"query": query_text, "expected": expected_set or set()})

        i += 1

    return entries

def compute_metrics(retrieved: set[str], expected: set[str]) -> dict:
    tp = len(retrieved & expected)
    precision = tp / len(retrieved) if retrieved else 0.0
    recall    = tp / len(expected)  if expected  else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "tp":        tp,
        "retrieved": len(retrieved),
        "expected":  len(expected),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate VSM queries against expected results.")
    parser.add_argument("queries_file", nargs="?", default="queries.txt",
                        help="Path to the evaluation file (default: queries.txt)")
    parser.add_argument("--alpha", type=float, default=0.005,
                        help="Cosine-similarity threshold (default: 0.005)")
    parser.add_argument("--speeches_dir", default="Speeches",
                        help="Directory of speech documents (default: Speeches)")
    parser.add_argument("--index_dir", default="index",
                        help="Directory for saved index (default: index)")
    parser.add_argument("--stopwords", default="stopwords.txt",
                        help="Path to stopwords file (default: stopwords.txt)")
    args = parser.parse_args()

    stopwords = load_stopwords(args.stopwords)

    if os.path.exists(args.index_dir):
        print(f"[INFO] Loading saved index from '{args.index_dir}' …")
        doc_ids, tf_index, df_index = load_index(args.index_dir)
    else:
        print(f"[INFO] Building index from '{args.speeches_dir}' …")
        doc_ids, tf_index, df_index = build_index(args.speeches_dir, stopwords)
        save_index(tf_index, df_index, doc_ids, args.index_dir)

    print(f"[INFO] {len(doc_ids)} documents indexed, {len(df_index)} unique terms.\n")

    doc_vectors = compute_tfidf(tf_index, df_index, doc_ids)


    entries = parse_query_file(args.queries_file)
    print(f"[INFO] {len(entries)} queries loaded from '{args.queries_file}'\n")
    print("=" * 72)

    all_metrics = []

    for idx, entry in enumerate(entries, 1):
        query    = entry["query"]
        expected = entry["expected"]

        raw_results = process_query(
            query, stopwords, df_index, doc_vectors, doc_ids, alpha=args.alpha
        )

        # Extract  the doc IDs returned by your system
        retrieved = {str(doc_id) for doc_id, _score in raw_results}

        metrics = compute_metrics(retrieved, expected)
        all_metrics.append(metrics)

        print(f"Query {idx} : {query}")
        print(f"Expected  {metrics['expected']:>3} : {sorted(expected, key=lambda x: int(x) if x.isdigit() else x)}")
        print(f"Retrieved {metrics['retrieved']:>3}: {sorted(retrieved, key=lambda x: int(x) if x.isdigit() else x)}")

        only_in_expected  = expected  - retrieved
        only_in_retrieved = retrieved - expected
        if only_in_expected:
            print(f"  Missed : {sorted(only_in_expected, key=lambda x: int(x) if x.isdigit() else x)}")
        if only_in_retrieved:
            print(f"  Extra : {sorted(only_in_retrieved, key=lambda x: int(x) if x.isdigit() else x)}")

        print(f"  Precision={metrics['precision']:.3f}  "
              f"Recall={metrics['recall']:.3f}  "
              f"F1={metrics['f1']:.3f}  "
              f"(TP={metrics['tp']})\n")



    if all_metrics:
        avg_p  = sum(m["precision"] for m in all_metrics) / len(all_metrics)
        avg_r  = sum(m["recall"]    for m in all_metrics) / len(all_metrics)
        avg_f1 = sum(m["f1"]        for m in all_metrics) / len(all_metrics)

        print(f"\nSUMMARY  ({len(all_metrics)} queries)")
        print(f"  Avg Precision : {avg_p:.3f}")
        print(f"  Avg Recall    : {avg_r:.3f}")
        print(f"  Avg F1        : {avg_f1:.3f}")

        perfect = sum(1 for m in all_metrics if m["f1"] == 1.0)
        print(f"  Perfect match : {perfect}/{len(all_metrics)} queries")


if __name__ == "__main__":
    main()