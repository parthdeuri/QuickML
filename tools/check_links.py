#!/usr/bin/env python3
"""Check every card's reference URL (HTTP status). Usage: python tools/check_links.py"""
import concurrent.futures as cf
import glob
import sys
import urllib.error
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
UA = {"User-Agent": "Mozilla/5.0 (QuickML link check)"}


def check(url):
    base = url.split("#")[0]
    for method in ("HEAD", "GET"):
        req = urllib.request.Request(base, headers=UA, method=method)
        try:
            with urllib.request.urlopen(req, timeout=25) as r:
                return url, r.status, r.geturl()
        except urllib.error.HTTPError as e:
            if method == "GET" or e.code not in (403, 405, 400, 501):
                return url, e.code, ""
        except Exception as e:  # noqa: BLE001
            if method == "GET":
                return url, f"ERR {type(e).__name__}", ""
    return url, "?", ""


def main():
    urls = {}
    for f in sorted(glob.glob(str(ROOT / "content" / "problems" / "*.yaml"))):
        d = yaml.safe_load(open(f, encoding="utf-8"))
        urls.setdefault(d["url"], []).append(d["slug"])
    bad = 0
    with cf.ThreadPoolExecutor(3) as ex:
        for url, status, final in ex.map(check, urls):
            ok = status == 200
            if not ok:
                bad += 1
                print(f"{status}  {url}  <- {', '.join(urls[url])}")
    print(f"{len(urls)} urls checked, {bad} not OK")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
