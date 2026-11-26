#!/usr/bin/env python3
"""
Download and unpack the Paderborn bearing archives used by the pipeline.

    python data/download_paderborn.py            # healthy + real damages (17 archives)
    python data/download_paderborn.py --all      # also artificial damages
    python data/download_paderborn.py --dest D:/paderborn

Each archive is a few hundred MB; the default set is several GB in total.
Unpacking .rar needs one of: 7-Zip (Windows: 7z.exe on PATH or in Program Files),
`unar` (macOS: brew install unar) or `unrar` (Linux: sudo apt install unrar).
Already downloaded or unpacked bearings are skipped, so the script can be re-run.
"""
import argparse
import os
import shutil
import subprocess
import sys
import ssl
import urllib.error
import urllib.request

# Use the operating system's certificate store (Windows/macOS) instead of
# Python's bundled one. Antivirus HTTPS scanning or a campus/company proxy adds
# its own root certificate to the OS store only, and Python then fails with
# "self-signed certificate in certificate chain". Verification stays ON.
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    truststore = None

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))
from paderborn import BASE_URL, HEALTHY, REAL_OUTER, REAL_INNER, ART_OUTER, ART_INNER  # noqa: E402


def find_extractor():
    for exe in ("7z", "7za", "unar", "unrar"):
        if shutil.which(exe):
            return exe
    for p in (r"C:\Program Files\7-Zip\7z.exe", r"C:\Program Files (x86)\7-Zip\7z.exe"):
        if os.path.exists(p):
            return p
    return None


def extract(exe, archive, dest):
    name = os.path.basename(exe).lower()
    if name.startswith("7z"):
        cmd = [exe, "x", "-y", f"-o{dest}", archive]
    elif name == "unar":
        cmd = [exe, "-f", "-o", dest, archive]
    else:
        cmd = [exe, "x", "-o+", archive, dest + os.sep]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def download(url, path):
    try:
        _download_urllib(url, path)
    except urllib.error.URLError as e:
        if not isinstance(e.reason, ssl.SSLCertVerificationError):
            raise
        # curl.exe (built into Windows 10/11) and macOS curl verify against the
        # OS certificate store, so they succeed where Python's bundle fails.
        curl = shutil.which("curl")
        if curl is None:
            raise SystemExit(
                "SSL certificate check failed. Run `pip install truststore` and try again.") from e
        print("  Python SSL check failed; retrying with curl (OS certificate store)")
        tmp = path + ".part"
        subprocess.run([curl, "-L", "--fail", "-C", "-", "-o", tmp, url], check=True)
        os.replace(tmp, path)


def _download_urllib(url, path):
    tmp = path + ".part"
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        got = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if total:
                print(f"\r  {got >> 20} / {total >> 20} MB", end="", flush=True)
    print()
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "paderborn"))
    ap.add_argument("--all", action="store_true", help="include artificial damages")
    ap.add_argument("--keep-rar", action="store_true")
    a = ap.parse_args()

    codes = HEALTHY + REAL_OUTER + REAL_INNER + (ART_OUTER + ART_INNER if a.all else [])
    os.makedirs(a.dest, exist_ok=True)
    exe = find_extractor()
    if exe is None:
        print("No .rar extractor found (7-Zip, unar or unrar). Archives will be downloaded only.")

    for code in codes:
        folder = os.path.join(a.dest, code)
        if os.path.isdir(folder) and any(f.endswith(".mat") for f in os.listdir(folder)):
            print(f"{code}: already unpacked")
            continue
        rar = os.path.join(a.dest, f"{code}.rar")
        if not os.path.exists(rar):
            print(f"{code}: downloading {BASE_URL}{code}.rar")
            download(f"{BASE_URL}{code}.rar", rar)
        if exe:
            print(f"{code}: unpacking")
            extract(exe, rar, a.dest)
            if not a.keep_rar:
                os.remove(rar)
    print("done:", a.dest)


if __name__ == "__main__":
    main()
