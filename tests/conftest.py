import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
CANONICAL = REPO / "outputs" / "canonical_world_obs20_pred60.npz"
SPLITS_FILE = REPO / "splits" / "scene_splits.json"

# Model scripts do `import dd_data` / `import dd_split`, which resolves when
# they run as `python scripts/<name>.py`; make it resolve under pytest too.
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def load_script(name: str):
    """Import a scripts/*.py module even when the filename starts with a digit."""
    path = SCRIPTS / name
    mod_name = "script_" + name.replace(".py", "").replace(".", "_")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO


# ---------------------------------------------------------------------------
# Shared eval-population contract (benchmark v2)
# ---------------------------------------------------------------------------
# Every benchmark model must be scored on the EXACT same population: the agents
# of outputs/canonical_world_obs20_pred60.npz, partitioned by the frozen scene
# split (splits/scene_splits.json), with the same per-timestep fut_mask. These
# helpers derive that ground-truth population directly from the committed
# artifacts so a model's evaluation can be gated against it. Used by
# tests/test_eval_population_contract.py and by each new model's test suite.

def canonical_split_populations(canonical=CANONICAL, splits_file=SPLITS_FILE):
    """Ground-truth {split: {"agents": int, "valid_steps": int}} from the
    committed canonical dataset + frozen split. `valid_steps` is the number of
    observed future timesteps (fut_mask) — the masked-metric denominator."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import dd_split

    d = np.load(canonical, allow_pickle=True)
    mapping = dd_split.load_splits(splits_file)
    sse = d["seq_start_end"]
    scene_id = d["scene_id"].astype(object)
    fut_mask = d["fut_mask"].astype(bool)
    pops = {s: {"agents": 0, "valid_steps": 0} for s in ("train", "val", "test")}
    for i, (a, b) in enumerate(sse):
        split = mapping.get(str(scene_id[i]))
        if split not in pops:
            continue
        pops[split]["agents"] += int(b - a)
        pops[split]["valid_steps"] += int(fut_mask[a:b].sum())
    return pops


def assert_eval_population(n_val_agents, n_test_agents, pops=None):
    """Hard gate: a model's val/test evaluation must cover exactly the canonical
    per-split agent population — no scene-centric survivor filter, no private
    split, no 'all agents in scene'. Off-by-anything means the numbers compare
    to nothing."""
    pops = pops or canonical_split_populations()
    assert n_val_agents == pops["val"]["agents"], (
        f"val eval population {n_val_agents} != canonical {pops['val']['agents']}")
    assert n_test_agents == pops["test"]["agents"], (
        f"test eval population {n_test_agents} != canonical {pops['test']['agents']}")
