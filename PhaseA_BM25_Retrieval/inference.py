"""
testset_inference.py
--------------------
Adapted from original testset_inference.py (PISA + JSONL baselines)
→ Now uses: pt.terrier.Retriever + reads content from xml.gz files

Usage:
    python testset_inference.py
    python testset_inference.py --add_contents
    python testset_inference.py --topk 100 --k1 1.1 --b 0.3
"""

import pyterrier as pt
if not pt.java.started():
    pt.java.init()

import pandas as pd
import json, os, gzip, re
import xml.etree.ElementTree as ET
import argparse

from grid_search import load_index


# ------------------------------------------------------------------ #
#  CONFIGURE                                                          #
# ------------------------------------------------------------------ #

XMLGZ_FOLDER = r"C:\Users\malla\Desktop\thesis\BioASQ\BioASQ13B-master\pubmed_xmls"
TESTSET_FILE = r"C:\Users\malla\Desktop\thesis\BioASQ\BioASQ13B-master\PhaseA_BM25_Retrieval\BioASQ-task13bPhaseA-testset1.json"
OUTPUT_FILE  = r"C:\Users\malla\Desktop\thesis\BioASQ\BioASQ13B-master\results\testset_output.jsonl"

DEFAULT_K1           = 1.1
DEFAULT_B            = 0.3
DEFAULT_TOPK         = 100
DEFAULT_ADD_CONTENTS = False

# ------------------------------------------------------------------ #


def clean_query(text: str) -> str:
    """
    Clean query text for Terrier query parser.
    Removes special characters that crash the parser:
    apostrophes, quotes, question marks, parentheses, hyphens etc.

    Example:
        'What proportion of colorectal cancer cases (CMS) groups?'
        → 'what proportion of colorectal cancer cases  cms  groups'
    """
    text = text.lower()
    text = text.replace("'", " ")    # apostrophe
    text = text.replace('"', " ")    # double quote
    text = text.replace("?", " ")    # question mark  ← main culprit
    text = text.replace("(", " ")    # parentheses    ← main culprit
    text = text.replace(")", " ")
    text = text.replace("/", " ")    # slash
    text = text.replace("-", " ")    # hyphen
    text = text.replace(":", " ")    # colon
    text = text.replace(";", " ")    # semicolon
    text = text.replace(",", " ")    # comma
    text = text.replace(".", " ")    # period
    text = text.replace("[", " ")    # square brackets
    text = text.replace("]", " ")
    text = text.replace("%", " ")    # percent
    text = text.replace("+", " ")    # plus
    text = text.replace("=", " ")    # equals
    text = re.sub(r'\s+', ' ', text) # collapse multiple spaces
    return text.strip()


def get_queries(filepath: str):
    """
    Reads BioASQ JSON test file.
    Applies clean_query() to every query body before returning.

    Returns:
        queries      : [ {qid, query}, ... ]   ← query is cleaned
        queryid2text : { qid: original_text }  ← original kept for output
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = json.load(f)

    questions    = content["questions"]

    # clean_query applied here — fixes all special char crashes
    queries      = [{"qid": q["id"], "query": clean_query(q["body"])} for q in questions]
    queryid2text = {q["id"]: q["body"] for q in questions}   # keep original for output

    print(f"Loaded {len(queries)} test queries from '{filepath}'.")
    return queries, queryid2text


def add_content(docs: dict) -> dict:
    """
    Mirrors original add_content(baseline, docs).
    Reads title + abstract from xml.gz files for each retrieved PMID.
    """
    lst_docs    = [doc["id"] for doc_list in docs.values() for doc in doc_list]
    docs_needed = set(lst_docs)
    total       = len(docs_needed)
    doc_content = {}

    print(f"\nLoading content for {total:,} PMIDs from xml.gz files...")

    xmlgz_files = sorted(
        os.path.join(XMLGZ_FOLDER, f)
        for f in os.listdir(XMLGZ_FOLDER)
        if f.endswith(".xml.gz")
    )

    found = 0
    for file_path in xmlgz_files:
        if not docs_needed:
            break

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

    print(f"\n  Loaded content for {found}/{total} PMIDs.")

    results_with_content = {}
    for qid, doc_list in docs.items():
        enriched = []
        for doc in doc_list:
            pid = doc["id"]
            if pid not in doc_content:
                print(f"  error: PMID {pid} not found in xml.gz files")
                text = ""
            else:
                text = doc_content[pid]
            enriched.append({**doc, "text": text})
        results_with_content[qid] = enriched

    return results_with_content


def run(testset_file, output_file, topk, k1, b, add_contents):
    """Main logic."""

    # 1. Load + clean queries
    queries, queryid2text = get_queries(testset_file)

    # 2. Load index
    index = load_index()
    print(f"\nBM25 params: k1={k1}, b={b}, topk={topk}\n")

    # 3. BM25 retrieval
    bm25 = pt.terrier.Retriever(
        index,
        wmodel="BM25",
        num_results=topk,
        controls={"bm25.k_1": k1, "bm25.b": b},
    )

    questions_df = pd.DataFrame(queries)
    raw          = bm25.transform(questions_df)

    # Build results dict
    results = {
        qid: [{"id": row["docno"]} for _, row in group.iterrows()]
        for qid, group in raw.groupby("qid")
    }

    # 4. Optionally add content
    if add_contents:
        results = add_content(results)

    # 5. Write output jsonl
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for qid, docs in results.items():
            out = {
                "id":         qid,
                "query_text": queryid2text.get(qid, ""),   # original text in output
                "bm25":       docs,
            }
            f.write(json.dumps(out) + "\n")

    print(f"\nOutput written to '{output_file}'.")
    print(f"  {len(results)} queries written.")


def main():
    parser = argparse.ArgumentParser(description="BioASQ Testset BM25 Inference")
    parser.add_argument("--testset_file",  default=TESTSET_FILE)
    parser.add_argument("--output_file",   default=OUTPUT_FILE)
    parser.add_argument("--topk",          type=int,   default=DEFAULT_TOPK)
    parser.add_argument("--k1",            type=float, default=DEFAULT_K1)
    parser.add_argument("--b",             type=float, default=DEFAULT_B)
    parser.add_argument("--add_contents",  action="store_true", default=DEFAULT_ADD_CONTENTS)
    args = parser.parse_args()

    print("=" * 60)
    print("Testset Inference")
    print("=" * 60)

    run(
        testset_file = args.testset_file,
        output_file  = args.output_file,
        topk         = args.topk,
        k1           = args.k1,
        b            = args.b,
        add_contents = args.add_contents,
    )


if __name__ == "__main__":
    main()