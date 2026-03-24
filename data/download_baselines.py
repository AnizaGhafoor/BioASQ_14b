import urllib.request
import re
import gzip
import os
import shutil

BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
DOWNLOAD_FOLDER = "pubmed_xmls"

os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

# -----------------------
# GET FILE LIST
# -----------------------

print("Getting PubMed file list...")

page = urllib.request.urlopen(BASE_URL).read().decode("utf-8")
files = re.findall(r'href="(pubmed\d{2}n\d{4}\.xml\.gz)"', page)
links = [BASE_URL + f for f in files]

print("Total files found:", len(links))

# -----------------------
# DOWNLOAD FILES
# -----------------------

print("\nStarting downloads...")

def is_file_valid(path):
    return os.path.exists(path) and os.path.getsize(path) > 1024  # at least 1KB

for link in links:
    filename = link.split("/")[-1]
    filepath = os.path.join(DOWNLOAD_FOLDER, filename)

    # Skip if already downloaded and valid
    if is_file_valid(filepath):
        try:
            with gzip.open(filepath, "rb") as f:
                f.read(1)
            print("Already downloaded:", filename)
            continue
        except (OSError, EOFError):
            print("Corrupted file, re-downloading:", filename)
            os.remove(filepath)

    print("Downloading:", filename)

    for attempt in range(3):  # retry up to 3 times
        try:
            with urllib.request.urlopen(link) as response, open(filepath, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
            print("Saved:", filename)
            break
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")

print("\nAll downloads completed.")