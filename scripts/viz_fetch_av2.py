"""Fetch a sample of Argoverse 2 motion-forecasting scenarios (Western comparison).

Argoverse 2 (Wilson et al., NeurIPS 2021 Datasets & Benchmarks) is public on S3
with no registration (licence CC BY-NC-SA 4.0: academic, non-commercial use).
Each scenario is 11 s at 10 Hz (110 frames), the same frame rate as IDD-3D, so
our 2 s -> 6 s models run on it without resampling. Each scenario directory has
`scenario_<id>.parquet` (tracks) and `log_map_archive_<id>.json` (vector map).

Only the val split is sampled; downloads land in data_external/av2/val/<id>/
(gitignored). Files are parsed as data only.

Usage:
    python scripts/viz_fetch_av2.py --n 120
"""

from __future__ import annotations

import argparse
import re
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BUCKET = "https://s3.amazonaws.com/argoverse"
PREFIX = "datasets/av2/motion-forecasting/val/"


def list_scenarios(n: int) -> list[str]:
    """First n scenario ids under the val prefix (S3 ListObjectsV2, delimiter '/')."""
    ids, token = [], None
    while len(ids) < n:
        q = {"list-type": "2", "prefix": PREFIX, "delimiter": "/", "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        xml = urllib.request.urlopen(f"{BUCKET}?{urllib.parse.urlencode(q)}", timeout=60).read().decode()
        ids += re.findall(r"<Prefix>" + re.escape(PREFIX) + r"([0-9a-f-]{36})/</Prefix>", xml)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
        if not m:
            break
        token = m.group(1)
    return ids[:n]


def fetch(sid: str, out: Path) -> None:
    d = out / sid
    d.mkdir(parents=True, exist_ok=True)
    for name in (f"scenario_{sid}.parquet", f"log_map_archive_{sid}.json"):
        f = d / name
        if not f.exists():
            urllib.request.urlretrieve(f"{BUCKET}/{PREFIX}{sid}/{name}", f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--out", default=str(REPO / "data_external" / "av2" / "val"))
    args = ap.parse_args()
    out = Path(args.out)
    ids = list_scenarios(args.n)
    for i, sid in enumerate(ids, 1):
        fetch(sid, out)
        if i % 20 == 0:
            print(f"{i}/{len(ids)}")
    print(f"fetched {len(ids)} scenarios -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
