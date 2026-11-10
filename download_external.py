"""Download IDSSE from its original CC BY 4.0 Figshare deposit; verify MD5."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request
import subprocess

ROOT = Path(__file__).resolve().parent / "external_data" / "idsse"
API = "https://api.figshare.com/v2/articles/28196177"


def fetch(entry):
    target = ROOT / entry["name"]
    if target.exists() and target.stat().st_size == entry["size"]:
        if hashlib.md5(target.read_bytes()).hexdigest() == entry["computed_md5"]:
            return target.name + " verified existing"
    temp = target.with_suffix(".download")
    for attempt in range(3):
        try:
            command = ["curl.exe", "--fail", "--location", "--silent", "--show-error",
                       "--connect-timeout", "20", "--max-time", "240", "--speed-limit", "10240",
                       "--speed-time", "30", "--continue-at", "-", "--output", str(temp), entry["download_url"]]
            subprocess.run(command, check=True)
            if temp.stat().st_size != entry["size"] or hashlib.md5(temp.read_bytes()).hexdigest() != entry["computed_md5"]:
                raise ValueError("Size/checksum mismatch")
            temp.replace(target)
            return target.name + " verified downloaded"
        except Exception:
            if attempt == 2:
                raise


if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(API, timeout=60) as stream:
        article = json.load(stream)
    if article["license"]["url"].rstrip("/") != "https://creativecommons.org/licenses/by/4.0":
        raise ValueError("Unexpected licence; review before download")
    (ROOT / "source_manifest.json").write_text(json.dumps(article, indent=2), encoding="utf-8")
    files = [f for f in article["files"] if "positions_raw" in f["name"] or "matchinformation" in f["name"]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(fetch, f) for f in files]
        for future in concurrent.futures.as_completed(futures):
            print(future.result(), flush=True)
