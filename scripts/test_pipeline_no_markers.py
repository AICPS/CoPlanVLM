#!/usr/bin/env python3
"""test_pipeline_no_markers.py — the offline pipeline with the red-X obstacle marks REMOVED.

The ablation arm for measuring how much the red markers actually help the VLM. It runs the SAME
pipeline as the baseline harnesses in every respect except two, both of them the manipulation:

  * the overlay image carries only BLUE DOTS at every grid point (map_gen.render_grid_points_map),
    with no red X marks; and
  * no prompt text anywhere mentions red markers or blocked points — the overlay description is
    prompt_gen._NOMARK_OVERLAY, the CoT scaffold is COT_BLOCKS_NOMARK (red-marker sub-step removed,
    remaining sub-steps re-lettered), and the reply schema's per-step descriptions come from
    COT_FIELDS_NOMARK.

Everything upstream is untouched: CLIPSeg still segments, the occupancy grid is still built and
inflated, and A* / coverage still plan against that inflated grid. This ablates what the VLM is
TOLD, not where the robots can actually drive — so the effect shows up as the VLM choosing cells the
planner then has to project or drop, not as collisions.

    --mode sim   runs scripts/test_pipeline.py's pipeline      (Gazebo overhead.png + poses.json)
    --mode real  runs scripts/test_pipeline_real.py's pipeline (AVL_* lab images, --scene)

HOW IT WORKS (worth knowing before debugging a surprising run): both harnesses look up
``generate_prompt`` and ``route_schema`` as module globals in test_pipeline's namespace, so this
script rebinds those two names to their no-marker twins and then calls the harness's own main().
That is deliberate — it means the ablation runs the identical planning loop, debug writers and goal
separation as the baseline, with no second copy to drift out of sync. The rebinds are process-global,
which is safe here only because this script is a standalone entry point that runs one mode and exits.

Prerequisites:
    colcon build --symlink-install --packages-select coplan_vlm
    source install/setup.bash

Usage (from workspace root):
    python3 src/CoPlanVLM/scripts/test_pipeline_no_markers.py --mode sim \\
        --prompt "Send both robots to the person in the purple shirt"

    python3 src/CoPlanVLM/scripts/test_pipeline_no_markers.py --mode real --scene AVL_3 \\
        --prompt "Have donnie circle around the box on the left and then go to the chair"

Any other argument the underlying harness accepts (--planner, --model, --temperature, --cot/--no-cot,
--data, --camera, --out, and --scene in real mode) is forwarded unchanged.

Output goes to debug/ablation_nomarkers_{sim,real}/ by default — a DIFFERENT directory from the
baseline runs (debug/offline_test_sim, debug/offline_test_real), so the two arms of the study never
overwrite each other. Same file set as the baseline harness.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_pipeline as tp             # noqa: E402
import test_pipeline_real as tpr       # noqa: E402
from node_Executive_API.prompt_gen import (   # noqa: E402
    generate_prompt_nomark, route_schema_nomark)

# Default output directory per mode. Deliberately not the baseline's, so an ablation run can never
# clobber the results it is meant to be compared against.
_OUT_DIRS = {"sim": "debug/ablation_nomarkers_sim", "real": "debug/ablation_nomarkers_real"}


def _install_ablation_hooks() -> None:
    """Rebind test_pipeline's prompt + schema builders to their no-red-marker twins.

    THIS is what makes the run an ablation. Both harnesses resolve these two names from
    test_pipeline's module globals at call time (test_pipeline.main -> generate_prompt,
    test_pipeline._call_planner_vlm -> route_schema, and test_pipeline_real.main -> tp.generate_prompt),
    so rebinding here covers both modes and leaves every other stage of the pipeline alone.

    The twins take the same arguments and return the same shapes, so nothing downstream can tell the
    difference — which is the point: the only variable is the marker condition.
    """
    tp.generate_prompt = generate_prompt_nomark
    tp.route_schema = route_schema_nomark


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False)          # -h is forwarded to the harness so its own options are listed too
    parser.add_argument("--mode", choices=("sim", "real"), default="sim",
                        help="Which harness to ablate: sim = test_pipeline.py (Gazebo image), "
                             "real = test_pipeline_real.py (AVL_* lab images). Default: sim.")
    parser.add_argument("-h", "--help", action="store_true", dest="want_help")
    args, passthrough = parser.parse_known_args()

    harness = {"sim": tp, "real": tpr}[args.mode]

    if args.want_help:
        print(__doc__)
        print("─" * 70)
        print(f"Options forwarded to the --mode {args.mode} harness:")
        passthrough = ["--help"]
    else:
        # Force the ablation's own output directory unless the caller picked one, so the baseline
        # results stay intact.
        if not any(a == "--out" or a.startswith("--out=") for a in passthrough):
            passthrough += ["--out", _OUT_DIRS[args.mode]]
        # Force the points overlay. generate_prompt_nomark ignores the argument when building the
        # prompt, but the harness also prints it and hands it to debug_io.save_vlm_selections; left
        # at the default the console would claim "overlay: marked_obs" for a run that has no red X
        # marks at all. points and marked_obs draw vlm_selections.png identically, so this changes
        # the log line and nothing else.
        if not any(a == "--map-overlay" or a.startswith("--map-overlay=") for a in passthrough):
            passthrough += ["--map-overlay", "points"]
        _install_ablation_hooks()
        print(f"[ablation] no red markers — mode={args.mode}, "
              f"prompt+schema from generate_prompt_nomark / route_schema_nomark")

    # The harness parses sys.argv itself, so hand it exactly the arguments meant for it.
    sys.argv = [f"{sys.argv[0]} --mode {args.mode}"] + passthrough
    harness.main()


if __name__ == "__main__":
    main()
