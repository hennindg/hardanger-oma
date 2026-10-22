"""
Download Hardanger Bridge data files from the official repository and unzip them.

Dataset: "Wind and Acceleration Data from the Hardanger Bridge", Fenerci et al., CC-BY 4.0,
DOI 10.21400/5ng8980s (Sikt NVA, https://nva.sikt.no/registration/01977e2c97f8-09ea4ff5-fd74-48c0-bd99-790e3d8c19ac).

For each file in FILES:
  1. ask the repository API for a temporary download link (the "filelink" endpoint),
  2. download the zip file in chunks to DOWNLOAD_FOLDER (skipped if it is already there with
     the right size),
  3. check the size against the repository record,
  4. unzip it into its own folder in DATA_ROOT, named after the zip file.

Only the Python standard library is used (no extra packages).

How to use:
  python -m tools.download_data
"""

from pathlib import Path
import json
import time
import urllib.request
import zipfile
from hardanger.paths import DATA_FOLDER


# ============================ SETTINGS ============================

PUBLICATION = "01977e2c97f8-09ea4ff5-fd74-48c0-bd99-790e3d8c19ac"
API = "https://api.nva.unit.no/publication"

DATA_ROOT = DATA_FOLDER
DOWNLOAD_FOLDER = DATA_ROOT / "downloads"

# Files to download (names as in the repository): all LowFreq (2 Hz) data sets used in the project,
# 27 zip files, about 4.3 GB (September 2014, storms, max wind, low-wind events, full year 2016).
FILES = [
    "LowFreq_2014_9.zip", "Events_StormUrd_LowFreq.zip",
    "Events_StormNina_LowFreq.zip", "Events_StormOle_LowFreq.zip", "Events_StormRoar_LowFreq.zip",
    "Events_StormTor_LowFreq.zip", "Events_Max_Wind_LowFreq.zip",
    "Events_Low_Wind_2013_12_08_LowFreq.zip", "Events_Low_Wind_2015_12_16_LowFreq.zip",
    "Events_Low_Wind_2015_12_19_LowFreq.zip", "Events_Low_Wind_2016_01_17_LowFreq.zip",
    "Events_Low_Wind_2016_01_18_LowFreq.zip", "Events_Low_Wind_2016_01_30_LowFreq.zip",
    "Events_Low_Wind_2016_02_02_LowFreq.zip", "Events_Low_Wind_2016_03_30_LowFreq.zip",
] + [f"LowFreq_2016_{month}.zip" for month in range(1, 13)]

CHUNK_BYTES = 8 * 1024 * 1024     # download in 8 MB pieces

# ==================================================================


def get_json(url):
    """Read a JSON answer from the repository API."""
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def list_repository_files():
    """All files of the publication: {name: (file identifier, size in bytes)}."""
    record = get_json(f"{API}/{PUBLICATION}")
    files = {}
    for artifact in record["associatedArtifacts"]:
        if artifact.get("name"):
            files[artifact["name"]] = (artifact["identifier"], artifact.get("size"))
    return files


def download(name, identifier, size):
    """Download one file to DOWNLOAD_FOLDER. Returns the local path."""
    target = DOWNLOAD_FOLDER / name
    if target.exists() and target.stat().st_size == size:
        print(f"   already downloaded")
        return target

    # Temporary signed link to the file storage
    link = get_json(f"{API}/{PUBLICATION}/filelink/{identifier}")["id"]
    partial = target.with_suffix(".part")
    t0 = time.time()
    with urllib.request.urlopen(link, timeout=120) as response, open(partial, "wb") as out:
        received = 0
        while True:
            chunk = response.read(CHUNK_BYTES)
            if not chunk:
                break
            out.write(chunk)
            received += len(chunk)
    if received != size:
        raise RuntimeError(f"size mismatch: got {received} bytes, expected {size}")
    partial.replace(target)
    print(f"   downloaded {received / 1e6:.1f} MB in {time.time() - t0:.0f} s")
    return target


def unzip(path):
    """Unzip into DATA_ROOT / <zip name without .zip>, unless that folder already has .mat files."""
    folder = DATA_ROOT / path.stem
    if folder.exists() and any(folder.rglob("*.mat")):
        print(f"   already unzipped in {folder.name}")
        return folder
    with zipfile.ZipFile(path) as z:
        z.extractall(folder)
    n_mat = len(list(folder.rglob("*.mat")))
    print(f"   unzipped {n_mat} .mat files into {folder.name}")
    return folder


# --------------------------- Main program ---------------------------

DOWNLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
repository = list_repository_files()
total = sum(repository[name][1] for name in FILES)
print(f"{len(FILES)} files, {total / 1e9:.2f} GB in total\n")

for k, name in enumerate(FILES):
    identifier, size = repository[name]
    print(f"[{k + 1}/{len(FILES)}] {name} ({size / 1e6:.1f} MB)")
    try:
        path = download(name, identifier, size)
        unzip(path)
    except Exception as error:
        # One failed file should not stop the others; run the script again to retry
        print(f"   FAILED: {error}")

print("\nDone. Folders with .mat files:")
for name in FILES:
    folder = DATA_ROOT / Path(name).stem
    if folder.exists():
        for sub in sorted({p.parent for p in folder.rglob("*.mat")}):
            print(f"   {sub.relative_to(DATA_ROOT)}  ({len(list(sub.glob('*.mat')))} recordings)")
