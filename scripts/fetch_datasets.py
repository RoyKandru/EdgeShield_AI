"""
Download the public datasets used to evaluate the EdgeShield detectors.

Datasets (all public; check each licence before redistribution):
  sms      UCI SMS Spam Collection (CC BY 4.0)            -> data/external/sms/SMSSpamCollection
  url      HF ealvaradob/phishing-dataset urls.json (Apache-2.0) -> data/external/url/urls.json
  network  CIC-IDS2017 per-flow CSVs (UNB, research use)  -> data/external/network/*.csv

SHA-256 of every file is written to data/external/MANIFEST.json so a reported
metric can always be tied to the exact bytes it was computed on.

Usage: python scripts/fetch_datasets.py [sms] [url] [network]   (default: all)
"""
import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "external"
HF = "https://huggingface.co/datasets"

SOURCES = {
    "sms": [
        ("https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
         "sms/sms_spam_collection.zip", True),
    ],
    "url": [
        (f"{HF}/ealvaradob/phishing-dataset/resolve/main/urls.json", "url/urls.json", False),
    ],
    "network": [
        (f"{HF}/Mireu-Lab/CIC-IDS/resolve/main/CSV/{n}", f"network/{n}", False)
        for n in (
            "Monday-WorkingHours.pcap_ISCX.csv",
            "Tuesday-WorkingHours.pcap_ISCX.csv",
            "Wednesday-workingHours.pcap_ISCX.csv",
            "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
            "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
            "Friday-WorkingHours-Morning.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        )
    ],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, rel: str, unzip: bool, manifest: dict) -> None:
    dest = OUT / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"downloading {url}")
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as exc:  # 404 for files that do not exist upstream
            print(f"  SKIP ({exc})")
            return
    files = [dest]
    if unzip:
        with zipfile.ZipFile(dest) as z:
            z.extractall(dest.parent)
            files += [dest.parent / n for n in z.namelist() if not n.endswith("/")]
    for f in files:
        manifest[str(f.relative_to(OUT)).replace("\\", "/")] = {
            "sha256": sha256(f), "bytes": f.stat().st_size, "source": url,
        }


def main() -> None:
    wanted = sys.argv[1:] or list(SOURCES)
    mpath = OUT / "MANIFEST.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    for name in wanted:
        for url, rel, unzip in SOURCES[name]:
            fetch(url, rel, unzip, manifest)
    OUT.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"manifest: {mpath}")


if __name__ == "__main__":
    main()
