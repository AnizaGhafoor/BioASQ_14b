"""
grid_search.py
--------------
Fully automatic — mirrors original grid_search.py logic exactly.

Original flow:
    1. get_queries()  -> reads .jsonl training file, groups by baseline
    2. load_index()   -> loads PISA index per baseline
    3. grid search    -> tries all k1/b combinations
    4. get_metrics()  -> recall/map/ndcg
    5. write_results() + calculate_average_evaluation()

This version:
    1. get_queries()  -> reads BioASQ training JSON AUTOMATICALLY
                         extracts PMIDs from document URLs
                         filters to only queries whose PMIDs exist in index
    2. load_index()   -> loads PyTerrier index (Windows compatible)
    3. grid search    -> identical to original
    4. get_metrics()  -> identical metrics
    5. write_results() + calculate_average_evaluation() -> identical

Usage:
    python grid_search.py
"""

import pyterrier as pt
if not pt.java.started():
    pt.java.init()

import pandas as pd
from itertools import product
from collections import defaultdict
from ranx import Qrels, Run, evaluate
import json, os, sys, re


# ------------------------------------------------------------------ #
#  CONFIGURE — only change these paths                                #
# ------------------------------------------------------------------ #

INDEX_PATH    = r"C:\Users\malla\Desktop\thesis\BioASQ\BioASQ13B-master\data\indexes\pubmed_index"
TRAINING_FILE = r"PhaseA_BM25_Retrieval/trainining14b.json"
RESULTS_PATH  = r"C:\Users\malla\Desktop\thesis\BioASQ\BioASQ13B-master\results\grid_search"

NUM_RESULTS   = 1000   # top-k docs to retrieve per query (original: 1000)
MAX_QUERIES   = None   # None = use ALL matched queries
                       # set e.g. 50 for a quick test run

# ------------------------------------------------------------------ #


def clean_query(text: str) -> str:
    """
    Clean query text for Terrier query parser.
    Removes ALL special characters that crash the parser.
    Original text is preserved separately for output.

    Example:
        'What is the function of 6sRNA in bacteria?'
        -> 'what is the function of 6srna in bacteria'

        'Which disease is caused by (CMS) groups?'
        -> 'which disease is caused by  cms  groups'
    """
    text = text.lower()
    for ch in ["'", '"', "?", "(", ")", "/", "-", ":", ";", ",", ".", "[", "]", "%", "+", "="]:
        text = text.replace(ch, " ")
    text = re.sub(r'\s+', ' ', text)   # collapse multiple spaces
    return text.strip()


def load_index():
    """
    Mirrors original load_index(baseline).
    Original : PisaIndex(path, text_field='text', threads=32)
    This file: pt.IndexFactory.of(path) — Windows compatible
    """
    if not os.path.exists(INDEX_PATH):
        print(f"[ERROR]: Index not found at '{INDEX_PATH}'.")
        print("Run create_indexes.py first.")
        sys.exit(-1)

    index = pt.IndexFactory.of(INDEX_PATH)
    stats = index.getCollectionStatistics()
    print(f"Index loaded from '{INDEX_PATH}'.")
    print(f"  Documents : {stats.getNumberOfDocuments():,}")
    print(f"  Terms     : {stats.getNumberOfUniqueTerms():,}")
    return index


def get_indexed_pmids(index) -> set:
    """
    Build set of all PMIDs stored in the index.
    Used to filter training queries automatically —
    only keeps queries whose relevant docs exist in our index.
    """
    print("\nReading all PMIDs from index...")
    meta  = index.getMetaIndex()
    total = index.getCollectionStatistics().getNumberOfDocuments()

    pmids = set()
    for i in range(total):
        pmids.add(meta.getItem('docno', i))

    print(f"  {len(pmids):,} unique PMIDs in index.")
    return pmids


def get_queries(training_file: str, indexed_pmids: set):
    """
    Automatic version of original get_queries(filename).

    Original reads .jsonl line by line:
        question["id"], question["documents"], question["body"], question["baseline"]

    This reads BioASQ JSON:
        { "questions": [ {"id":..., "body":..., "documents": [urls...]} ] }

    Key steps:
        1. Extract PMID from URL: "http://.../pubmed/12345678" -> "12345678"
        2. Filter: only keep queries where at least 1 relevant PMID is in index
        3. Apply clean_query() to avoid Terrier parser crashes
        4. Build queries list + qrels_dict + queryid2text automatically
    """
    print(f"\nLoading training queries from:")
    print(f"  '{training_file}'\n")

    with open(training_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_questions = data.get("questions", [])
    print(f"  Total questions in file : {len(all_questions):,}")

    queries      = []   # [ {qid, query}, ... ]
    qrels_dict   = {}   # { qid: { pmid: 1, ... } }
    queryid2text = {}   # { qid: original_text }
    skipped      = 0

    for question in all_questions:
        qid  = question["id"]
        body = question["body"]
        docs = question.get("documents", [])

        # Extract PMID from URL (mirrors: doc["id"] in original)
        # "http://www.ncbi.nlm.nih.gov/pubmed/12345678" -> "12345678"
        pmids = [d.split("/")[-1] for d in docs if d]

        # Keep only PMIDs that are in our index
        matched_pmids = [p for p in pmids if p in indexed_pmids]

        if not matched_pmids:
            skipped += 1
            continue   # none of this query's relevant docs are in our index

        # Apply clean_query() — prevents Terrier parser crashes
        # mirrors: {"qid": qid, "query": query.lower()}
        queries.append({"qid": qid, "query": clean_query(body)})

        # Build qrels (mirrors: { qid: { did: 1 for did in doc_ids } })
        qrels_dict[qid]   = {p: 1 for p in matched_pmids}
        queryid2text[qid] = body   # keep original text for output

        if MAX_QUERIES and len(queries) >= MAX_QUERIES:
            break

    print(f"  Matched queries (PMIDs in index): {len(queries):,}")
    print(f"  Skipped queries (no match)      : {skipped:,}")

    if not queries:
        print("\n[ERROR] No matching queries found.")
        print("  Your training file PMIDs may not overlap with your indexed files.")
        print("  Try indexing more xml.gz files, or set MAX_QUERIES = 10 and re-run.")
        sys.exit(-1)

    return queries, qrels_dict, queryid2text


def run_bm25(index, queries_df: pd.DataFrame, k1: float, b: float) -> dict:
    """
    Mirrors original BM25 retrieval:
        bm25 = index.bm25(k1=k1, b=b, num_results=1000, threads=32)
        run_dict = { qid: { docno: score } ... }
    """
    bm25 = pt.terrier.Retriever(
        index,
        wmodel="BM25",
        num_results=NUM_RESULTS,
        controls={"bm25.k_1": k1, "bm25.b": b},
    )

    results = bm25.transform(queries_df)

    if results.empty:
        return {}

    return {
        qid: {row["docno"]: row["score"] for _, row in group.iterrows()}
        for qid, group in results.groupby("qid")
    }


def get_metrics(qrels_dict: dict, run_dict: dict) -> dict:
    """
    Mirrors original get_metrics() exactly — same metrics and cutoffs.
    """
    qrels = Qrels(qrels_dict)
    run   = Run(run_dict)

    metrics = [
        "recall@1000", "recall@200", "recall@100", "recall@10",
        "map@1000",    "map@200",    "map@100",    "map@10",
        "ndcg@1000",   "ndcg@200",   "ndcg@100",   "ndcg@10",
    ]
    return evaluate(qrels, run, metrics, make_comparable=True)


def write_results(results: dict, filename: str = "bm25_gridsearch.json"):
    """Mirrors original write_results(baseline, results, results_path)."""
    os.makedirs(RESULTS_PATH, exist_ok=True)
    out_file = os.path.join(RESULTS_PATH, filename)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to '{out_file}'.")


def calculate_average_evaluation():
    """
    Mirrors original calculate_average_evaluation() exactly.
    Averages all JSON files in RESULTS_PATH -> bm25_avg.json
    """
    sum_scores = defaultdict(lambda: defaultdict(float))
    num_files  = 0

    for filename in os.listdir(RESULTS_PATH):
        if filename == "bm25_avg.json":
            continue
        fpath = os.path.join(RESULTS_PATH, filename)
        if not os.path.isfile(fpath):
            continue

        num_files += 1
        with open(fpath, "r") as f:
            results = json.load(f)
            for params, metrics in results.items():
                for metric, score in metrics.items():
                    sum_scores[params][metric] += score

    if num_files == 0:
        print("[WARNING] No result files found to average.")
        return

    avg_results = {
        params: {metric: val / num_files for metric, val in metrics.items()}
        for params, metrics in sum_scores.items()
    }

    avg_file = os.path.join(RESULTS_PATH, "bm25_avg.json")
    with open(avg_file, "w") as f:
        json.dump(avg_results, f, indent=2)
    print(f"Average results -> '{avg_file}'.")


def print_top_results(evaluation: dict, metric: str = "ndcg@10", top_n: int = 5):
    """Print top N param settings sorted by metric."""
    sorted_params = sorted(
        evaluation.items(),
        key=lambda x: x[1].get(metric, 0),
        reverse=True,
    )

    print(f"\nTop {top_n} parameter settings by '{metric}':")
    print(f"  {'Params':<25} {'ndcg@10':>9} {'recall@10':>10} {'map@10':>8} {'ndcg@100':>10}")
    print("  " + "-" * 68)
    for rank, (params, metrics) in enumerate(sorted_params[:top_n], 1):
        print(
            f"  {rank}. {params:<22}"
            f"  {metrics.get('ndcg@10',   0):>9.4f}"
            f"  {metrics.get('recall@10', 0):>10.4f}"
            f"  {metrics.get('map@10',    0):>8.4f}"
            f"  {metrics.get('ndcg@100',  0):>10.4f}"
        )

    best_params = sorted_params[0][0]
    best_score  = sorted_params[0][1].get(metric, 0)
    print(f"\nBest params: {best_params}  ->  {metric} = {best_score:.4f}")
    print(f"\nCopy these into search_all.py and testset_inference.py:")
    try:
        k1_val, b_val = best_params.strip("()").split(",")
        print(f"  K1 = {k1_val.strip()}")
        print(f"  B  = {b_val.strip()}")
    except Exception:
        pass


def main():
    print("=" * 60)
    print("BM25 Grid Search — Fully Automatic")
    print("=" * 60)

    # 1. Load index
    index = load_index()

    # 2. Get all PMIDs in index
    indexed_pmids = get_indexed_pmids(index)

    # 3. Auto-load queries + qrels from training file
    queries, qrels_dict, queryid2text = get_queries(TRAINING_FILE, indexed_pmids)

    # 4. Build queries DataFrame
    queries_df = pd.DataFrame(queries)

    # 5. BM25 parameter grid — identical to original
    k1_lst       = [i / 10 for i in range(1, 13)]   # 0.1 -> 1.2  (12 values)
    b_lst        = [i / 10 for i in range(1, 11)]   # 0.1 -> 1.0  (10 values)
    combinations = list(product(k1_lst, b_lst))      # 120 combinations

    print(f"\nGrid: {len(combinations)} combinations  x  {len(queries)} queries\n")

    # 6. Grid search loop — mirrors original exactly
    evaluation = {}
    total      = len(combinations)

    for i, (k1, b) in enumerate(combinations, 1):
        print(f"  [{i:>3}/{total}]  k1={k1:.1f}  b={b:.1f}", end="\r")

        run_dict = run_bm25(index, queries_df, k1=k1, b=b)

        if not run_dict:
            print(f"\n  [WARNING] No results for k1={k1}, b={b} — skipping.")
            continue

        evaluation[str((k1, b))] = get_metrics(qrels_dict, run_dict)

    print(f"\n\nGrid search complete. {len(evaluation)} combinations evaluated.")

    # 7. Save + average (mirrors original)
    write_results(evaluation)
    calculate_average_evaluation()

    # 8. Show best params
    print_top_results(evaluation, metric="ndcg@10")


if __name__ == "__main__":
    main()