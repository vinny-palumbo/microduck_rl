"""Score exactly one completed navigation run against separately recorded simulator truth.

Usage: python -m mjlab_microduck.sim.mission_eval TRUTH.jsonl RUN_DIRECTORY

This is a post-run evaluator, never a controller input. Absolute wall timestamps
in events.jsonl bind mission.json to its own truth interval on the same host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from mjlab_microduck.sim.navigation_eval import (
    APARTMENT_SCENES,
    ArrivalEvaluator,
    MAX_SAMPLE_GAP,
    SCHEMA,
    arrival_conditions,
)

START_EVENTS = {"live_session_started": "live_session_finished", "mission_started": "mission_finished"}


def mission_interval(run: Path) -> tuple[dict, float, float, dict]:
    """Reject incomplete, ambiguous or mismatched artifacts instead of guessing an interval."""
    mission_path = run / "mission.json" if run.is_dir() else run
    result = json.loads(mission_path.read_text(encoding="utf-8"))
    events_path = mission_path.with_name("events.jsonl")
    boundaries = []
    voice_fragments = []
    initial_goal = None
    starts = 0
    capture_count = 0
    event_digest = hashlib.sha256()
    with events_path.open("rb") as stream:
        for raw in stream:
            event_digest.update(raw)
            event = json.loads(raw)
            kind = event.get("event")
            if kind in START_EVENTS or kind in START_EVENTS.values():
                boundaries.append(event)
            if kind in START_EVENTS:
                initial_goal = event.get("goal")
            if kind == "navigation_started":
                starts += 1
            if kind == "voice_transcript":
                voice_fragments.append(str(event.get("text", "")))
            if kind in {"live_observation", "observation"}:
                capture_count += 1
    if len(boundaries) != 2 or START_EVENTS.get(boundaries[0]["event"]) != boundaries[1]["event"]:
        raise ValueError("run must contain exactly one matching mission start and finish")
    start, finish = boundaries
    if finish.get("result") != result:
        raise ValueError("mission.json differs from the recorded final mission result")
    first, last = float(start["at"]), float(finish["at"])
    if not math.isfinite(first) or not math.isfinite(last) or not first < last:
        raise ValueError("mission start/end timestamps must be finite and increasing")
    goal = result.get("goal", start.get("goal"))
    if not isinstance(goal, str) or "kitchen" not in goal.casefold():
        raise ValueError("this apartment evaluator only scores an explicit kitchen goal")
    provenance = {
        "mission_json": str(mission_path.resolve()),
        "mission_sha256": hashlib.sha256(mission_path.read_bytes()).hexdigest(),
        "events_jsonl": str(events_path.resolve()),
        "events_sha256": event_digest.hexdigest(),
        "initial_goal": initial_goal,
        "voice_transcript": "".join(voice_fragments),
        "navigation_started_events": starts,
        "recorded_observations": capture_count,
    }
    return result, first, last, provenance


def score_mission(truth_log: Path, run: Path, duck: int = 0) -> dict:
    """Use only this mission's samples, requiring continuous coverage through its stop."""
    result, start, end, provenance = mission_interval(run)
    evaluator = ArrivalEvaluator()
    first_sample = last_sample = verdict = None
    previous_wall = previous_sim = None
    max_wall_gap = max_sim_gap = path_length = 0.0
    contacts_known = contact_steps = contact_count = fallen_samples = 0
    max_penetration = max_speed = 0.0
    obstacles = set()
    problems = set()
    selected_digest = hashlib.sha256()
    with truth_log.open("rb") as stream:
        header = next(stream)
        metadata = json.loads(header)
        if (metadata.get("schema") != SCHEMA or metadata.get("scene") not in APARTMENT_SCENES
                or metadata.get("goal") != "kitchen"):
            raise ValueError("not a supported kitchen-navigation truth log")
        selected_digest.update(header)
        for raw in stream:
            if not raw.endswith(b"\n"):
                break
            sample = json.loads(raw)
            if sample.get("type") != "sample" or sample.get("duck") != duck:
                continue
            wall, sim = float(sample["recorded_at"]), float(sample["sim_time"])
            if not math.isfinite(wall) or not math.isfinite(sim):
                problems.add("nonfinite_truth_clock")
                continue
            if wall > end:
                break
            if wall < start:
                continue
            if previous_wall is not None and (wall <= previous_wall or sim <= previous_sim):
                problems.add("truth_clock_not_strictly_increasing")
            previous_wall, previous_sim = wall, sim
            selected_digest.update(raw)
            contacts = sample.get("contacts")
            if contacts is not None:
                contacts_known += 1
                contact_steps += contacts["contact_steps"]
                contact_count += contacts["contact_count"]
                max_penetration = max(max_penetration, contacts["max_penetration_m"])
                obstacles.update(contacts["obstacles"])
            if sample["released"] and (sample["up_alignment"] < 0.5 or sample["position_m"][2] < 0.06):
                fallen_samples += 1
            max_speed = max(max_speed, sample["speed_mps"])
            if last_sample is not None:
                wall_gap = wall - last_sample["recorded_at"]
                sim_gap = sim - last_sample["sim_time"]
                max_wall_gap = max(max_wall_gap, wall_gap)
                max_sim_gap = max(max_sim_gap, sim_gap)
                if wall_gap > MAX_SAMPLE_GAP or sim_gap > MAX_SAMPLE_GAP:
                    problems.add("truth_gap_within_mission")
                path_length += math.dist(sample["position_m"][:2], last_sample["position_m"][:2])
            else:
                first_sample = sample
            verdict = evaluator.update(sample)
            last_sample = sample
    if first_sample is None or last_sample is None or verdict is None:
        raise ValueError("truth log has no samples within the mission's wall-clock interval")
    start_gap = first_sample["recorded_at"] - start
    end_gap = end - last_sample["recorded_at"]
    if start_gap > MAX_SAMPLE_GAP:
        problems.add("truth_does_not_cover_mission_start")
    if end_gap > MAX_SAMPLE_GAP:
        problems.add("truth_does_not_cover_mission_finish")
    physical_arrival = verdict["arrived"] and not problems
    claimed_arrival = result.get("status") == "goal_observed"
    acknowledged_stop = result.get("stop", {}).get("completed") is True
    collision_evidence = contacts_known == evaluator.samples
    safety_verified = collision_evidence and contact_steps == 0 and fallen_samples == 0
    return {
        "schema": "microduck.navigation.mission-score.v1", "goal": "kitchen", "duck": duck,
        "success": physical_arrival and claimed_arrival and acknowledged_stop and safety_verified,
        "physical_arrival": physical_arrival,
        "planner_claimed_arrival": claimed_arrival,
        "final_stop_acknowledged": acknowledged_stop,
        "safety_verified": safety_verified,
        "mission_status": result.get("status"),
        "mission_reason": result.get("reason"),
        "interval": {"wall_start": start, "wall_end": end,
                     "first_sim_time": first_sample["sim_time"], "last_sim_time": last_sample["sim_time"]},
        "coverage": {"valid": not problems, "problems": sorted(problems),
                     "samples": evaluator.samples, "start_gap_s": start_gap, "end_gap_s": end_gap,
                     "max_wall_gap_s": max_wall_gap, "max_sim_gap_s": max_sim_gap},
        "initial_position_m": first_sample["position_m"],
        "initial_inside_kitchen": arrival_conditions(first_sample)["inside_kitchen"],
        "final_position_m": last_sample["position_m"],
        "travelled_xy_m": path_length,
        "net_displacement_xy_m": math.dist(first_sample["position_m"][:2], last_sample["position_m"][:2]),
        "final_conditions": verdict["conditions"], "final_dwell_s": verdict["dwell_s"],
        "safety": {"collision_evidence_available": collision_evidence,
                   "collision_free": contact_steps == 0 if collision_evidence else None,
                   "contact_steps": contact_steps if collision_evidence else None,
                   "contact_count": contact_count if collision_evidence else None,
                   "max_penetration_m": max_penetration if collision_evidence else None,
                   "obstacles": sorted(obstacles), "fallen_samples": fallen_samples,
                   "max_speed_mps": max_speed,
                   "contact_boundary_note": "First truth window can include contacts just before "
                   "mission start; counted conservatively. No post-finish truth samples are used."},
        "ever_arrived_during_mission": evaluator.first_arrival is not None,
        "sources": {**provenance, "truth_jsonl": str(truth_log.resolve()),
                    "selected_truth_sha256": selected_digest.hexdigest()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("truth_log", type=Path)
    parser.add_argument("run", type=Path, help="run directory, or its mission.json")
    parser.add_argument("--duck", type=int, default=0)
    args = parser.parse_args()
    report = score_mission(args.truth_log, args.run, args.duck)
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if report["success"] else 1)


if __name__ == "__main__":
    main()
