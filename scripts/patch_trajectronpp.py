"""Compatibility patches for the upstream Trajectron++ codebase (2020-era)
so it runs on Python 3.12 / NumPy 2.x / PyTorch 2.x.

Idempotent: safe to run multiple times. Applied patches:
1. `from collections import Sequence` -> `collections.abc` (Python 3.10+).
2. Removed NumPy scalar aliases (np.float / np.int / np.bool / np.object).
3. tensorboardX -> torch.utils.tensorboard.
4. torch.load(..., weights_only=False) in model_registrar (PyTorch 2.6
   default flipped to True; the checkpoint stores an nn.ModuleDict).
5. pack_padded_sequence lengths must be on CPU (PyTorch >= 1.7).

Usage:
    python scripts/patch_trajectronpp.py [--repo third_party/trajectron-plus-plus]
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
]

# (file, old, new, skip_if_present) — skip_if_present makes each patch
# idempotent even when `new` still contains the `old` substring.
SPECIFIC = [
    (
        "trajectron/environment/data_structures.py",
        "from collections import Sequence, OrderedDict",
        "from collections.abc import Sequence\nfrom collections import OrderedDict",
        "from collections.abc import Sequence",
    ),
    (
        "trajectron/train.py",
        "from tensorboardX import SummaryWriter",
        "from torch.utils.tensorboard import SummaryWriter",
        "from torch.utils.tensorboard import SummaryWriter",
    ),
    (
        "trajectron/model/model_registrar.py",
        "self.model_dict = torch.load(save_path, map_location=self.device)",
        "self.model_dict = torch.load(save_path, map_location=self.device, weights_only=False)",
        "weights_only=False",
    ),
    (
        "trajectron/model/model_utils.py",
        "packed_seqs = rnn.pack_padded_sequence(inp, seq_lens, batch_first=True, enforce_sorted=False)",
        "packed_seqs = rnn.pack_padded_sequence(inp, torch.as_tensor(seq_lens).cpu(), batch_first=True, enforce_sorted=False)",
        "torch.as_tensor(seq_lens).cpu()",
    ),
    (
        # ncls has no Windows cp312 wheel; it is only needed by MultiNode
        # (online robot-node streaming), which this project does not use.
        "trajectron/environment/node.py",
        "from ncls import NCLS",
        "try:\n    from ncls import NCLS\nexcept ImportError:  # optional; only used by MultiNode\n    NCLS = None",
        "except ImportError:  # optional; only used by MultiNode",
    ),
    (
        # torch's SummaryWriter (unlike old tensorboardX) rejects plain lists.
        "trajectron/evaluation/evaluation.py",
        "log_writer.add_histogram(f\"{node_type.name}/{namespace}/{metric}\", metric_batch_error, curr_iter)",
        "log_writer.add_histogram(f\"{node_type.name}/{namespace}/{metric}\", np.asarray(metric_batch_error), curr_iter)",
        "np.asarray(metric_batch_error)",
    ),
]


def patch_file(path: Path, replacements: list[tuple[str, str, str]]) -> int:
    text = path.read_text(encoding="utf-8")
    n = 0
    for old, new, guard in replacements:
        if guard in text:
            continue  # already applied
        if old in text:
            text = text.replace(old, new)
            n += 1
    if n:
        path.write_text(text, encoding="utf-8")
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="third_party/trajectron-plus-plus")
    args = ap.parse_args()
    repo = Path(args.repo)
    if not repo.is_dir():
        raise SystemExit(f"Trajectron++ repo not found at {repo}. Clone it first:\n"
                         f"  git clone --depth 1 https://github.com/StanfordASL/Trajectron-plus-plus.git {repo}")

    total = 0
    for rel, old, new, guard in SPECIFIC:
        p = repo / rel
        if p.exists():
            total += patch_file(p, [(old, new, guard)])

    for p in (repo / "trajectron").rglob("*.py"):
        text = p.read_text(encoding="utf-8")
        new_text = text
        for rx, repl in ALIAS_RE:
            new_text = rx.sub(repl, new_text)
        if new_text != text:
            p.write_text(new_text, encoding="utf-8")
            total += 1

    print(f"applied patches to {total} file(s) (0 means already patched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
