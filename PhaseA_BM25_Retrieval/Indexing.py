# import pyterrier as pt
# import os
# import glob

# # Initialize PyTerrieri
# if not pt.started():
#     pt.init()

# def load_txt_files(txt_files: list):
#     for file_path in txt_files:
#         docno = os.path.splitext(os.path.basename(file_path))[0]
#         with open(file_path, "r", encoding="utf-8") as f:
#             text = f.read().strip()
#         if not text:
#             print(f"[WARNING] '{file_path}' is empty — skipping.")
#             continue
#         yield {"docno": docno, "text": text}

# def main():
#     txt_folder = "documents"
#     #index_path = "data/indexes/txt_index"
#     index_path = os.path.join(os.getcwd(), "data", "indexes", "txt_index")

#     txt_files = sorted(glob.glob(os.path.join(txt_folder, "*.txt")))
#     if not txt_files:
#         print(f"No .txt files found in '{txt_folder}'.")
#         return

#     print(f"Found {len(txt_files)} file(s):")
#     for f in txt_files:
#         print(f"  {f}")

#     # Use the standard IterDictIndexer (cross-platform, replaces Pisa)
#     os.makedirs(index_path, exist_ok=True)
#     indexer = pt.IterDictIndexer(index_path, meta=['docno'], verbose=True)

#     # Index documents
#     indexref = indexer.index(load_txt_files(txt_files))
#     print("\nIndexing completed!")

#     # Open the index
#     index = pt.IndexFactory.of(indexref)

#     # Test search
#     pipeline = pt.BatchRetrieve(index, wmodel="BM25")
#     results = pipeline.search("test query")
#     print(results[["docno", "score"]].head())

# if __name__ == "__main__":
#     main()

"""
create_indexes.py
-----------------
Adapted from original create_indexes.py (PISA + JSONL)
→ Now uses: PyTerrier IterDictIndexer + XML.GZ PubMed baseline files

Original : reads .jsonl files, uses PisaIndex (Linux only)
This file: reads .xml.gz files, uses pt.IterDictIndexer (Windows compatible)

Usage:
    python create_indexes.py
"""

import pyterrier as pt
if not pt.java.started():
    pt.java.init()

import os
import glob
import gzip
import time
import xml.etree.ElementTree as ET


# ------------------------------------------------------------------ #
#  CONFIGURE                                                          #
# ------------------------------------------------------------------ #

XMLGZ_FOLDER = os.path.join(os.getcwd(), "pubmed_xmls")          # your xml.gz folder
INDEX_PATH   = os.path.join(os.getcwd(), "data", "indexes", "pubmed_index")

TEST_MODE    = True    # True  = only index first 10 files
                       # False = index ALL files (full baseline ~2-4 hrs)

# ------------------------------------------------------------------ #


def get_files(folder: str, test_mode: bool) -> list:
    """
    Original: glob.glob(folder, '*.jsonl')
    Now     : glob.glob(folder, '*.xml.gz')
    """
    files = sorted(glob.glob(os.path.join(folder, "*.xml.gz")))

    if not files:
        raise FileNotFoundError(
            f"No .xml.gz files found in '{folder}'.\n"
            f"Check that XMLGZ_FOLDER points to your pubmed_xmls directory."
        )

    if test_mode:
        files = files[:10]
        print(f"[TEST MODE] Using first {len(files)} files only.\n")
    else:
        print(f"Found {len(files)} .xml.gz files (full baseline).\n")

    total_mb = sum(os.path.getsize(f) for f in files) / (1024 * 1024)
    print(f"Total compressed size: {total_mb:.1f} MB")
    return files


def extract_text(element, tag: str) -> str:
    """Safely extract text from XML element — returns '' if missing."""
    node = element.find(tag)
    if node is not None and node.text:
        return node.text.strip()
    return ""


def parse_xmlgz(file_path: str, file_num: int, total_files: int):
    """
    Generator — parses one .xml.gz and yields document dicts.

    Original load_collection() read from .jsonl:
        pub["pmid"], pub["title"], pub["abstract"]

    This reads from XML:
        .//PMID, .//ArticleTitle, .//AbstractText, .//MeshHeading
    """
    fname = os.path.basename(file_path)
    print(f"  [{file_num:>4}/{total_files}] Parsing {fname} ...", end=" ", flush=True)
    t0    = time.time()
    count = 0

    try:
        with gzip.open(file_path, "rb") as gz:
            tree = ET.parse(gz)
            root = tree.getroot()

        # Mirrors original deduplication: latest_docs[pub["pmid"]] = {...}
        latest_docs = {}

        for article in root.findall(".//PubmedArticle"):

            # PMID → docno (mirrors: "docno": pub["pmid"])
            pmid_node = article.find(".//PMID")
            if pmid_node is None or not pmid_node.text:
                continue
            pmid = pmid_node.text.strip()

            # Title (mirrors: pub["title"])
            title = extract_text(article, ".//ArticleTitle")

            # Abstract — may have multiple nodes (mirrors: pub["abstract"])
            abstract_nodes = article.findall(".//AbstractText")
            abstract = " ".join(
                node.text.strip() for node in abstract_nodes if node.text
            )

            # MeSH terms — extra medical vocabulary (not in original, bonus)
            mesh_terms = [
                d.text.strip()
                for d in article.findall(".//MeshHeading/DescriptorName")
                if d.text
            ]
            mesh_str = " ".join(mesh_terms)

            # Combine into text field (mirrors: " ".join([pub["title"], pub["abstract"]]))
            text = " ".join(filter(None, [title, abstract, mesh_str]))
            if not text:
                continue

            # Deduplication: last entry for same PMID wins
            latest_docs[pmid] = {
                "docno": pmid,
                "text":  text,
            }
            count += 1

        yield from latest_docs.values()

    except Exception as e:
        print(f"\n  [ERROR] Failed to parse '{fname}': {e}")
        return

    elapsed = time.time() - t0
    print(f"{count:>7,} articles  ({elapsed:.1f}s)")


def load_collection(files: list):
    """
    Master generator across all files — mirrors original load_collection().
    Streams docs one at a time to avoid loading all into RAM.
    """
    total_files = len(files)
    seen        = {}   # global dedup across files

    print("\nParsing XML.GZ files:\n")
    for i, file_path in enumerate(files, 1):
        for doc in parse_xmlgz(file_path, i, total_files):
            seen[doc["docno"]] = doc   # overwrite if same PMID appears again

    print(f"\nTotal unique articles: {len(seen):,}\n")
    yield from seen.values()


def create_index(files: list):
    """
    Build PyTerrier index.

    Original: PisaIndex(index_path, text_field='text')
    Now     : pt.IterDictIndexer (works on Windows)
    """
    os.makedirs(INDEX_PATH, exist_ok=True)

    # Resume support — skip if index already built
    if os.path.exists(os.path.join(INDEX_PATH, "data.properties")):
        print(f"[INFO] Index already exists at '{INDEX_PATH}'.")
        print("       Delete the folder to re-index.\n")
        return pt.IndexFactory.of(INDEX_PATH)

    print(f"Building index at '{INDEX_PATH}' ...\n")
    t0 = time.time()

    indexer = pt.IterDictIndexer(
        INDEX_PATH,
        meta      = ["docno"],
        meta_lens = [20],
        verbose   = True,
    )

    indexref = indexer.index(load_collection(files))

    elapsed = time.time() - t0
    index   = pt.IndexFactory.of(indexref)
    stats   = index.getCollectionStatistics()

    print(f"\nIndexing complete in {elapsed/60:.1f} minutes.")
    print(f"  Documents : {stats.getNumberOfDocuments():,}")
    print(f"  Terms     : {stats.getNumberOfUniqueTerms():,}")
    print(f"  Saved to  : '{INDEX_PATH}'")
    return index


def main():
    print("=" * 60)
    print("PubMed XML.GZ Indexing")
    print("=" * 60)

    files = get_files(XMLGZ_FOLDER, TEST_MODE)
    index = create_index(files)

    # Quick sanity search
    print("\nTest search: 'COVID-19 treatment' ...")
    pipeline = pt.terrier.Retriever(index, wmodel="BM25", num_results=5)
    results  = pipeline.search("COVID-19 treatment")
    if results.empty:
        print("  [WARNING] No results — check your index.")
    else:
        print(results[["rank", "docno", "score"]].to_string(index=False))

    print("\nDone. Run grid_search.py next.")


if __name__ == "__main__":
    main()