"""Render scene bundles (scripts/viz_scene_bundle.py) into one self-contained interactive HTML viewer.

Plotly.js (a pinned build in outputs/viz/vendor/) and the bundles are inlined, so the file works
offline and can be opened in any modern browser. Rendering only: every number shown comes from the
bundles. Outputs stay under outputs/viz/ (git-ignored; development results, local files only).

Usage:
    python scripts/viz_render.py outputs/viz/bundles/idd_20220118105838_seq_8.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "scripts" / "viz_template" / "scene_viewer.html"
PLOTLY = REPO / "outputs" / "viz" / "vendor" / "plotly-2.35.3.min.js"
OUT = REPO / "outputs" / "viz" / "scenes"


def render(bundles: list[Path], out: Path, plotly: Path = PLOTLY) -> Path:
    if not plotly.exists():
        raise SystemExit(f"{plotly} missing: download https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.3/plotly.min.js there")
    js = plotly.read_text(encoding="utf-8")
    assert "</script" not in js.lower()
    data = json.dumps([json.loads(Path(b).read_text(encoding="utf-8")) for b in bundles], separators=(",", ":"))
    data = data.replace("</", "<\\/")                                   # never close the script element early
    html = TEMPLATE.read_text(encoding="utf-8")
    for marker in ("/*__PLOTLY__*/", "/*__BUNDLES__*/"):
        assert html.count(marker) == 1, marker
    html = "<!doctype html>\n<html lang=\"en\">\n" + html.replace("/*__PLOTLY__*/", js).replace("/*__BUNDLES__*/", data) + "\n</html>\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bundles", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out) if args.out else OUT / (Path(args.bundles[0]).stem + ".html")
    f = render([Path(b) for b in args.bundles], out)
    print(f"{f} ({f.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
