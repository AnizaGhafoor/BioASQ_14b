"""
search_all.py
-------------
Adapted from original search_all.py (PISA + JSONL baselines)
→ Now uses: pt.terrier.Retriever + reads content from xml.gz files
→ Automatically loads queries from BioASQ training JSON file
→ Applies clean_query() to avoid Terrier parser crashes

Usage:
    python search_all.py
"""

import pyterrier as pt
if not pt.java.started():
    pt.java.init()

import pandas as pd
import json, os, gzip, re
import xml.etree.ElementTree as ET

from grid_search import load_index, clean_query, get_indexed_pmids


# ------------------------------------------------------------------ #
#  CONFIGURE — only change these paths and BM25 params               #
# ------------------------------------------------------------------ #

XMLGZ_FOLDER  = r"C:\Users\malla\Desktop\thesis\BioASQ\BioASQ13B-master\pubmed_xmls"
TRAINING_FILE = r"PhaseA_BM25_Retrieval/trainining14b.json"
OUTPUT_FILE   = r"C:\Users\malla\Desktop\thesis\BioASQ\BioASQ13B-master\results\search_all_output.jsonl"

# Best BM25 params — copy from grid_search.py output
K1 = 1.1
B  = 0.3

NUM_RESULTS = 1000   # mirrors original: num_results=1000

# ------------------------------------------------------------------ #


def get_queries(training_file: str, indexed_pmids: set):
    """
    Automatically loads queries from BioASQ training JSON.
    Mirrors original get_queries() — filters to only matched PMIDs.
    Applies clean_query() to avoid Terrier parser crashes.

    Returns:
        queries      : [ {qid, query}, ... ]   <- cleaned for retrieval
        qrels_dict   : { qid: {pmid: 1} }      <- ground truth
        queryid2text : { qid: original_text }  <- original for output
    """
    print(f"\nLoading queries from '{training_file}' ...")

    with open(training_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_questions = data.get("questions", [])
    print(f"  Total questions: {len(all_questions):,}")

    queries      = []
    qrels_dict   = {}
    queryid2text = {}
    skipped      = 0

    for question in all_questions:
        qid   = question["id"]
        body  = question["body"]
        docs  = question.get("documents", [])

        # Extract PMID from URL: "http://.../pubmed/12345678" -> "12345678"
        pmids         = [d.split("/")[-1] for d in docs if d]
        matched_pmids = [p for p in pmids if p in indexed_pmids]

        if not matched_pmids:
            skipped += 1
            continue

        # clean_query() applied here — mirrors: query.lower() in original
        queries.append({"qid": qid, "query": clean_query(body)})
        qrels_dict[qid]   = {p: 1 for p in matched_pmids}
        queryid2text[qid] = body   # keep original for output

    print(f"  Matched queries : {len(queries):,}")
    print(f"  Skipped queries : {skipped:,}")
    return queries, qrels_dict, queryid2text


def add_content(docs: dict) -> dict:
    """
    Mirrors original add_content(baseline, docs).

    Original reads pubmed_baseline_X.jsonl:
        doc_content[pmid] = title + abstract

    Now reads xml.gz files directly from pubmed_xmls/ folder.
    Stops early once all needed PMIDs are found (mirrors original break).
    """
    lst_docs    = [doc["id"] for doc_list in docs.values() for doc in doc_list]
    docs_needed = set(lst_docs)
    total       = len(docs_needed)
    doc_content = {}

    print(f"\nLoading content for {total:,} unique PMIDs from xml.gz files...")

    xmlgz_files = sorted(
        os.path.join(XMLGZ_FOLDER, f)
        for f in os.listdir(XMLGZ_FOLDER)
        if f.endswith(".xml.gz")
    )

    found = 0
    for file_path in xmlgz_files:
        if not docs_needed:
            break   # all PMIDs found — stop early (mirrors original break)

        try:
            with gzip.open(file_path, "rb") as gz:
                tree = ET.parse(gz)
                root = tree.getroot()

            for article in root.findall(".//PubmedArticle"):
                pmid_node = article.find(".//PMID")
                if pmid_node is None or not pmid_node.text:
                    continue
                pmid = pmid_node.text.strip()

                if pmid not in docs_needed:
                    continue

                # Title + abstract (mirrors: " ".join([doc["title"], doc["abstract"]]))
                title_node = article.find(".//ArticleTitle")
                title      = title_node.text.strip() if (title_node is not None and title_node.text) else ""

                abstract_nodes = article.findall(".//AbstractText")
                abstract       = " ".join(n.text.strip() for n in abstract_nodes if n.text)

                doc_content[pmid] = " ".join(filter(None, [title, abstract]))
                docs_needed.discard(pmid)
                found += 1
                print(f"  {found}/{total}", end="\r")

                if not docs_needed:
                    break

        except Exception as e:
            print(f"\n  [WARNING] Could not parse '{file_path}': {e}")
            continue

    if docs_needed:
        print(f"\n  [WARNING] {len(docs_needed)} PMIDs not found in xml.gz files.")

    print(f"\n  Content loaded for {found}/{total} PMIDs.")

    # Attach content to each doc — mirrors: {**doc, **text}
    results_with_content = {}
    for qid, doc_list in docs.items():
        results_with_content[qid] = [
            {**doc, "text": doc_content.get(doc["id"], "")}
            for doc in doc_list
        ]

    return results_with_content


def main():
    print("=" * 60)
    print("Search All — BM25 Retrieval + Content")
    print("=" * 60)
    print(f"k1={K1}, b={B}, num_results={NUM_RESULTS}\n")

    # 1. Load index
    index = load_index()

    # 2. Get indexed PMIDs for query filtering
    indexed_pmids = get_indexed_pmids(index)

    # 3. Auto-load queries from training file
    queries, qrels_dict, queryid2text = get_queries(TRAINING_FILE, indexed_pmids)

    if not queries:
        print("[ERROR] No queries found. Check your training file and index.")
        return

    print(f"\nRunning BM25 retrieval for {len(queries)} queries...")

    # 4. BM25 retrieval (mirrors: bm25 = index.bm25(k1, b, num_results=1000))
    bm25 = pt.terrier.Retriever(
        index,
        wmodel="BM25",
        num_results=NUM_RESULTS,
        controls={"bm25.k_1": K1, "bm25.b": B},
    )

    queries_df = pd.DataFrame(queries)
    raw        = bm25.transform(queries_df)

    # Build results dict (mirrors: results = { qid: [{id, score}...] })
    results = {
        qid: [{"id": row["docno"], "score": row["score"]} for _, row in group.iterrows()]
        for qid, group in raw.groupby("qid")
    }

    # Filter empty ids (mirrors: [doc for doc in res if doc["id"]])
    bm25_rank = {
        qid: [doc for doc in doc_list if doc["id"]]
        for qid, doc_list in results.items()
    }

    # 5. Add content (mirrors: add_content(baseline, bm25_rank))
    bm25_with_content = add_content(bm25_rank)

    # 6. Write output jsonl (mirrors: f.write(json.dumps(out)))
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for qid, docs in bm25_with_content.items():
            out = {
                "id":         qid,
                "query_text": queryid2text.get(qid, ""),   # original text
                "bm25":       docs,
            }
            f.write(json.dumps(out) + "\n")

    print(f"\nOutput written to '{OUTPUT_FILE}'.")
    print(f"  {len(bm25_with_content)} queries  |  up to {NUM_RESULTS} docs each.")


if __name__ == "__main__":
    main()