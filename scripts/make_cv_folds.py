"""Scene-level 5-fold cross-validation splits (run once; commit the JSONs).

Every one of the 105 scenes is a test scene in exactly one fold, evaluated by
models that never trained on it. Scenes are ranked by md5 hash and dealt
round-robin into 5 folds of 21; for fold k the test scenes are fold k, val
(model selection + conformal calibration) is the 10 scenes of fold k+1 with the
lowest second hash, and the remaining 74 train -- the proportions of the frozen
benchmark split (73/10/22).
Same file format as splits/scene_splits.json, so every driver takes it via
--splits-file.

Usage:
    python scripts/make_cv_folds.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FROZEN = REPO / "splits" / "scene_splits.json"
OUT_DIR = REPO / "splits" / "cv_folds"
K = 5
N_VAL = 10                          # val scenes per fold (as in the frozen split)
SALT, VAL_SALT = "dreamdrive-cv5", "dreamdrive-cv5-val"


def bucket(scene_id: str, salt: str) -> float:
    h = hashlib.md5(f"{salt}:{scene_id}".encode("utf-8")).hexdigest()
    return int(h[:15], 16) / float(16 ** 15)


def make_folds(scene_ids: list[str]) -> list[dict]:
    # rank by hash, deal round-robin: balanced folds (21 scenes each for 105)
    ranked = sorted(scene_ids, key=lambda s: bucket(s, SALT))
    fold_of = {s: i % K for i, s in enumerate(ranked)}
    out = []
    for k in range(K):
        test = sorted(s for s in scene_ids if fold_of[s] == k)
        nxt = sorted((s for s in scene_ids if fold_of[s] == (k + 1) % K), key=lambda s: bucket(s, VAL_SALT))
        val = sorted(nxt[:N_VAL])
        train = sorted(s for s in scene_ids if s not in set(test) | set(val))
        out.append({"method": f"md5({SALT}:scene_id) rank, round-robin into {K} folds; test = fold k, "
                              f"val = {N_VAL} scenes of fold k+1 with the lowest md5({VAL_SALT}), train = rest",
                    "salt": SALT, "fold": k, "num_scenes": len(scene_ids),
                    "counts": {"train": len(train), "val": len(val), "test": len(test)},
                    "splits": {"train": train, "val": val, "test": test}})
    return out


def main() -> int:
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))["splits"]
    scenes = sorted(s for split in frozen.values() for s in split)
    folds = make_folds(scenes)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in folds:
        (OUT_DIR / f"fold{f['fold']}.json").write_text(json.dumps(f, indent=2) + "\n", encoding="utf-8")
        print(f"fold {f['fold']}: {f['counts']}")
    tested = [s for f in folds for s in f["splits"]["test"]]
    assert sorted(tested) == scenes, "every scene must be a test scene exactly once"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
