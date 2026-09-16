"""Independent simulator scoring; never import this module into a navigation planner.

The simulator writes an opt-in truth log. The scoring command runs after navigation,
and evaluates physical arrival separately from what a camera model claimed to see.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

SCHEMA = "microduck.navigation.truth.v1"
APARTMENT_SCENES = {"scene_navigation.xml", "scene_apartment_flat.xml", "scene_apartment.xml"}
# Interior wall faces from apartment.xml, inset by 20 cm to require the whole duck
# to cross the doorway. Coordinates are exclusively evaluator data.
KITCHEN_BOUNDS = (-3.74, -1.26, 0.76, 2.74)
DWELL_SECONDS = 1.0
MAX_SAMPLE_GAP = 0.5


def arrival_conditions(sample: dict) -> dict[str, bool]:
    """Arriving means standing still inside the room, with the policy supporting the body."""
    x, y, z = sample["position_m"]
    lo_x, hi_x, lo_y, hi_y = KITCHEN_BOUNDS
    finite = all(math.isfinite(float(value)) for value in (
        x, y, z, sample["up_alignment"], sample["speed_mps"], sample["angular_speed_rps"],
        sample["sim_time"],
    ))
    return {
        "finite": finite,
        "inside_kitchen": finite and lo_x <= x <= hi_x and lo_y <= y <= hi_y,
        "standing": finite and 0.08 <= z <= 0.20 and sample["up_alignment"] >= math.cos(math.radians(30)),
        "stopped": finite and sample["speed_mps"] <= 0.06 and sample["angular_speed_rps"] <= 0.4,
        "policy_support": bool(sample["released"] and sample["torque_on"]),
    }


class ArrivalEvaluator:
    """Require consecutive, fresh samples; a stopped log or a single pose cannot pass."""

    def __init__(self) -> None:
        self.previous_time: float | None = None
        self.inside_since: float | None = None
        self.first_arrival: float | None = None
        self.samples = 0

    def update(self, sample: dict) -> dict:
        now = float(sample["sim_time"])
        conditions = arrival_conditions(sample)
        valid_time = math.isfinite(now) and (
            self.previous_time is None or 0 < now - self.previous_time <= MAX_SAMPLE_GAP
        )
        if not valid_time or not all(conditions.values()):
            self.inside_since = None
        elif self.inside_since is None:
            self.inside_since = now
        dwell = 0.0 if self.inside_since is None else now - self.inside_since
        arrived = dwell + 1e-9 >= DWELL_SECONDS
        if arrived and self.first_arrival is None:
            self.first_arrival = now
        self.previous_time = now if math.isfinite(now) else None
        self.samples += 1
        return {"arrived": arrived, "dwell_s": dwell, "conditions": conditions}


class TruthLogger:
    """Writes pose and policy-support facts to a file that is outside the sensor protocol."""

    def __init__(self, path: Path, scene: Path, world) -> None:
        if scene.name not in APARTMENT_SCENES:
            raise ValueError("navigation evaluation requires an apartment scene")
        self.file = path.open("x", encoding="utf-8")
        self.world = world
        self.last_time = -math.inf
        self._write({
            "type": "metadata", "schema": SCHEMA, "scene": scene.name,
            "goal": "kitchen", "ducks": len(world.bodies), "recorded_at": time.time(),
            "kitchen_bounds_xy_m": KITCHEN_BOUNDS, "dwell_seconds": DWELL_SECONDS,
            "warning": "Evaluation ground truth. Do not supply to the planner.",
        })

    def _write(self, value: dict) -> None:
        self.file.write(json.dumps(value, allow_nan=False) + "\n")
        self.file.flush()

    def sample(self) -> None:
        import mujoco
        import numpy as np

        world = self.world
        samples = []
        with world.lock:
            now = float(world.data.time)
            if now - self.last_time < 0.099:
                return
            self.last_time = now
            for body in world.bodies:
                quaternion = world.data.qpos[body.trunk + 3:body.trunk + 7]
                rotation = np.empty(9)
                mujoco.mju_quat2Mat(rotation, quaternion)
                velocity = world.data.qvel[body.trunk_dof:body.trunk_dof + 6]
                samples.append({
                    "type": "sample", "duck": body.index, "sim_time": now,
                    "recorded_at": time.time(),
                    "position_m": world.data.qpos[body.trunk:body.trunk + 3].tolist(),
                    "quaternion_wxyz": quaternion.tolist(),
                    "up_alignment": float(rotation[8]),
                    "speed_mps": float(np.linalg.norm(velocity[:3])),
                    "angular_speed_rps": float(np.linalg.norm(velocity[3:])),
                    "released": body.released, "torque_on": body.torque_on,
                })
        for sample in samples:
            self._write(sample)

    def close(self) -> None:
        self.file.close()


def score_log(path: Path, duck: int = 0, start_time: float = 0.0,
              end_time: float = math.inf) -> dict:
    evaluator = ArrivalEvaluator()
    last = None
    verdict = None
    with path.open(encoding="utf-8") as stream:
        metadata = json.loads(next(stream))
        if metadata.get("schema") != SCHEMA or metadata.get("scene") not in APARTMENT_SCENES:
            raise ValueError("not an apartment navigation truth log")
        for line in stream:
            # The live simulator may currently be writing its final line. A complete
            # malformed record is still an error, not evidence that can be skipped.
            if not line.endswith("\n"):
                break
            sample = json.loads(line)
            if sample.get("type") != "sample" or sample.get("duck") != duck:
                continue
            if not start_time <= sample["sim_time"] <= end_time:
                continue
            verdict = evaluator.update(sample)
            last = sample
    if last is None or verdict is None:
        raise ValueError("no samples for the requested duck and time range")
    return {
        "schema": "microduck.navigation.score.v1", "goal": "kitchen", "duck": duck,
        "success": verdict["arrived"], "ever_arrived": evaluator.first_arrival is not None,
        "first_arrival_sim_time": evaluator.first_arrival,
        "samples": evaluator.samples, "final_sim_time": last["sim_time"],
        "final_position_m": last["position_m"], "final_conditions": verdict["conditions"],
        "final_dwell_s": verdict["dwell_s"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("truth_log", type=Path)
    parser.add_argument("--duck", type=int, default=0)
    parser.add_argument("--start-time", type=float, default=0.0, help="first simulator second to score")
    parser.add_argument("--end-time", type=float, default=math.inf, help="last simulator second to score")
    args = parser.parse_args()
    report = score_log(args.truth_log, args.duck, args.start_time, args.end_time)
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if report["success"] else 1)


if __name__ == "__main__":
    main()
