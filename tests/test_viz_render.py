"""Scene viewer rendering (scripts/viz_render.py): the HTML inlines the pinned Plotly.js build and the
bundles exactly once each, the bundle JSON round-trips, and nothing in the data can close the script
element early."""

import json
import re
from pathlib import Path

import pytest

from conftest import load_script

vr = load_script("viz_render.py")
needs_vendor = pytest.mark.skipif(not vr.PLOTLY.exists(), reason="Plotly.js vendor build not downloaded")


@needs_vendor
def test_the_html_inlines_plotly_and_the_bundle_safely(tmp_path):
    bundle = {"schema": 1, "scene": {"id": "x", "note": "a </script> inside the data"}, "windows": []}
    (tmp_path / "b.json").write_text(json.dumps(bundle), encoding="utf-8")
    html = vr.render([tmp_path / "b.json"], tmp_path / "v.html").read_text(encoding="utf-8")
    assert html.count("plotly.js v2.35.3") == 1 and "/*__PLOTLY__*/" not in html and "/*__BUNDLES__*/" not in html
    data = re.search(r"const BUNDLES = (.*?);</script>", html, re.S).group(1)
    assert "</script" not in data and json.loads(data.replace("<\\/", "</")) == [bundle]
    assert html.lower().count("</script>") == 3                         # plotly, bundles, viewer
