"""Compatibility patches for the upstream AgentFormer codebase (2021-era, pinned
to PyTorch ~1.5) so its *model* runs on Python 3.12 / NumPy 2.x / PyTorch 2.x.

Idempotent: safe to run multiple times. This project drives AgentFormer through
its model interface only (model.set_data / model.forward / model.inference), fed
canonical IDD-3D sequences directly by scripts/dd_agentformer.py — it never uses
AgentFormer's own data_generator/preprocessor. So the patches here target the
model chain (model/*.py) plus cheap, harmless cleanups to the data loader:

1. `_LinearWithBias` was removed from torch.nn.modules.linear (>= 1.9). The single
   use (MultiheadAttention-style out_proj) is a plain bias=True Linear.
2. `from torch.nn.functional import *` no longer re-exports typing names; the
   custom attention lib annotates with Tuple/Optional, so import them explicitly.
3. torchvision is imported at the top of model/map_encoder.py but only the resnet
   map encoders use it. IDD-3D is map-free (use_map=False), so make the import
   lazy — the map-free path needs neither torchvision nor a map CNN.
4. NumPy 2 removed the np.int / np.float / np.bool / np.object / np.str aliases.
5. cv2 (opencv) is imported at the top of the (unused) preprocessor; make it
   optional so the data package imports without opencv on a map-free install.

Still required as ordinary pip deps (documented in the README): easydict, glob2.

Usage:
    python scripts/patch_agentformer.py [--repo third_party/agentformer]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ALIAS_RE = [
    (re.compile(r"np\.float(?![\w.])"), "float"),
    (re.compile(r"np\.int(?![\w.])"), "int"),
    (re.compile(r"np\.bool(?![\w.])"), "bool"),
    (re.compile(r"np\.object(?![\w.])"), "object"),
    (re.compile(r"np\.str(?![\w.])"), "str"),
]

# (file, kind, a, b):
#   'replace'      -> text.replace(a, b) when a is present (idempotent because b
#                     never contains a here, so a is gone after the first pass);
#   'insert_after' -> insert b immediately after a, unless b is already present.
SPECIFIC = [
    ("model/agentformer_lib.py", "replace",
     "from torch.nn.modules.linear import Linear, _LinearWithBias",
     "from torch.nn.modules.linear import Linear"),
    ("model/agentformer_lib.py", "replace",
     "self.out_proj = _LinearWithBias(embed_dim, embed_dim)",
     "self.out_proj = Linear(embed_dim, embed_dim, bias=True)"),
    ("model/agentformer_lib.py", "insert_after",
     "from torch.nn.functional import *\n",
     "from typing import Optional, Tuple, List\n"),
    ("model/map_encoder.py", "replace",
     "import torch.nn.functional as F\nfrom torchvision import models\n",
     "import torch.nn.functional as F\n"),
    ("model/map_encoder.py", "insert_after",
     "        elif 'resnet' in model_id:\n",
     "            from torchvision import models  # lazy: only resnet map encoders need it\n"),
    ("data/preprocessor.py", "replace",
     "import torch, os, numpy as np, copy\nimport cv2",
     "import torch, os, numpy as np, copy\ntry:\n    import cv2\nexcept ImportError:  # optional; only the (unused) map loader needs it\n    cv2 = None"),
]


def apply_specific(repo: Path) -> int:
    n = 0
    for rel, kind, a, b in SPECIFIC:
        p = repo / rel
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        if kind == "replace":
            if a in text:
                text = text.replace(a, b)
                p.write_text(text, encoding="utf-8"); n += 1
        elif kind == "insert_after":
            if b not in text and a in text:
                text = text.replace(a, a + b, 1)
                p.write_text(text, encoding="utf-8"); n += 1
        else:
            raise ValueError(kind)
    return n


def apply_aliases(repo: Path) -> int:
    n = 0
    for sub in ("model", "utils", "data"):
        d = repo / sub
        if not d.is_dir():
            continue
        for p in d.rglob("*.py"):
            text = p.read_text(encoding="utf-8")
            new = text
            for rx, repl in ALIAS_RE:
                new = rx.sub(repl, new)
            if new != text:
                p.write_text(new, encoding="utf-8"); n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="third_party/agentformer")
    args = ap.parse_args()
    repo = Path(args.repo)
    if not repo.is_dir():
        raise SystemExit(
            f"AgentFormer repo not found at {repo}. Clone it first:\n"
            f"  git clone --depth 1 https://github.com/Khrylx/AgentFormer.git {repo}")

    total = apply_specific(repo) + apply_aliases(repo)
    print(f"applied patches to {total} file(s) (0 means already patched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
