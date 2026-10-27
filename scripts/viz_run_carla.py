"""One-shot CARLA visualization: start the server, replay the packs, compose videos.

Starts CARLA 0.9.16 off-screen, waits until it answers, renders every pair's
packs with scripts/viz_carla_replay.py, writes side-by-side videos with
scripts/viz_compose.py for both views, and shuts the server down.

Usage:
    python scripts/viz_run_carla.py \
        --pair idd_20220118103808_seq_18 av2_014586b7 \
        --pair idd_20220118112945_seq_22 av2_01340461
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PACKS = REPO / "outputs" / "viz" / "packs"
DEFAULT_CARLA = Path(r"C:\CARLA\CARLA_0.9.16\CarlaUE4.exe")


def check_directx_runtime() -> None:
    """UE4's launcher needs the legacy DirectX End-User Runtime (XInput 1.3, D3DX11...),
    which Windows 10/11 does not ship. Without it the launcher sits on a hidden
    "DirectX Runtime required" dialog and the server never comes up."""
    if not (Path(r"C:\Windows\System32") / "xinput1_3.dll").exists():
        raise SystemExit("CARLA needs the DirectX End-User Runtime (xinput1_3.dll missing). Install it "
                         "from an elevated terminal: winget install --id Microsoft.DirectX --exact")


def wait_for_server(port: int, timeout_s: float) -> None:
    import carla
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            c = carla.Client("localhost", port)
            c.set_timeout(5.0)
            print(f"CARLA server {c.get_server_version()} ready after {time.time() - t0:.0f} s")
            return
        except RuntimeError:
            time.sleep(5)
    raise SystemExit(f"CARLA did not answer on port {port} within {timeout_s:.0f} s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, action="append", required=True, metavar=("IDD_PACK", "AV2_PACK"))
    ap.add_argument("--carla", default=str(DEFAULT_CARLA))
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--quality", default="Epic", choices=["Low", "Epic"])
    args = ap.parse_args()

    check_directx_runtime()
    server = subprocess.Popen([args.carla, "-RenderOffScreen", f"-quality-level={args.quality}",
                               f"-carla-rpc-port={args.port}"])
    try:
        wait_for_server(args.port, 300.0)
        stems = sorted({s for pair in args.pair for s in pair})
        subprocess.run([sys.executable, str(REPO / "scripts" / "viz_carla_replay.py"),
                        *[str(PACKS / f"{s}.npz") for s in stems], "--port", str(args.port)],
                       check=True)
        for idd, av2 in args.pair:
            for view in ("chase", "bev"):
                subprocess.run([sys.executable, str(REPO / "scripts" / "viz_compose.py"),
                                "--pair", idd, av2, "--view", view], check=True)
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
        # CarlaUE4.exe launches a child shipping process; make sure it is gone too.
        subprocess.run(["taskkill", "/F", "/IM", "CarlaUE4-Win64-Shipping.exe"],
                       capture_output=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
