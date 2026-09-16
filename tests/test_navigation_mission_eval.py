"""Exact-run scoring must not borrow evidence from before or after the mission."""

import json

import pytest

from mjlab_microduck.sim.mission_eval import mission_interval, score_mission
from mjlab_microduck.sim.navigation_eval import SCHEMA


def write_run(tmp_path, start=1000.0, end=1002.5, status="goal_observed"):
    run = tmp_path / "run"
    run.mkdir()
    result = {"status": status, "goal": "Go to the kitchen", "goal_verified": False,
              "stop": {"completed": True}}
    (run / "mission.json").write_text(json.dumps(result))
    events = [
        {"at": start, "event": "live_session_started", "goal": None},
        {"at": start + 0.01, "event": "voice_transcript", "text": "Go to the kitchen."},
        {"at": start + 0.02, "event": "navigation_started", "goal": "kitchen"},
        {"at": end, "event": "live_session_finished", "result": result},
    ]
    (run / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    return run


def sample(t, inside=True):
    return {
        "type": "sample", "duck": 0, "recorded_at": 1000.0 + t, "sim_time": 10 + t,
        "position_m": [-1.5, 2.0, 0.125] if inside else [0, 0, 0.125],
        "up_alignment": 1.0, "speed_mps": 0.0, "angular_speed_rps": 0.0,
        "released": True, "torque_on": True,
        "contacts": {"physics_steps": 20, "contact_steps": 0, "contact_count": 0,
                     "max_penetration_m": 0.0, "obstacles": []},
    }


def write_truth(tmp_path, samples):
    path = tmp_path / "truth.jsonl"
    metadata = {"type": "metadata", "schema": SCHEMA, "scene": "scene_navigation.xml", "goal": "kitchen"}
    path.write_text("".join(json.dumps(record) + "\n" for record in [metadata, *samples]))
    return path


def test_scores_completed_mission_and_keeps_model_claim_separate(tmp_path):
    run = write_run(tmp_path)
    truth = write_truth(tmp_path, [sample(t / 10) for t in range(1, 25)])
    report = score_mission(truth, run / "mission.json")
    assert report["success"]
    assert report["physical_arrival"]
    assert report["safety_verified"]
    assert report["sources"]["initial_goal"] is None
    assert report["sources"]["voice_transcript"] == "Go to the kitchen."
    assert report["coverage"]["samples"] == 24
    assert json.loads((run / "mission.json").read_text())["goal_verified"] is False


def test_does_not_borrow_dwell_from_before_run_or_arrival_after_end(tmp_path):
    run = write_run(tmp_path, end=1000.6)
    truth = write_truth(tmp_path, [sample(t / 10) for t in range(-20, 30)])
    report = score_mission(truth, run)
    assert report["coverage"]["valid"]
    assert report["coverage"]["samples"] == 7
    assert not report["physical_arrival"]
    assert not report["success"]


def test_later_samples_cannot_change_finished_mission_result(tmp_path):
    run = write_run(tmp_path)
    samples = [sample(t / 10, inside=False) for t in range(1, 25)]
    truth = write_truth(tmp_path, samples)
    first = score_mission(truth, run)
    write_truth(tmp_path, samples + [sample(t / 10) for t in range(26, 60)])
    later = score_mission(truth, run)
    assert first == later
    assert not later["physical_arrival"]


@pytest.mark.parametrize("problem", ["start", "finish", "gap", "reversal"])
def test_insufficient_or_corrupt_time_coverage_cannot_pass(tmp_path, problem):
    run = write_run(tmp_path, end=1004)
    samples = [sample(t / 10) for t in range(1, 40)]
    if problem == "start":
        samples = samples[7:]
    elif problem == "finish":
        samples = samples[:-8]
    elif problem == "gap":
        samples = samples[:5] + samples[13:]
    else:
        samples[3], samples[4] = samples[4], samples[3]
    report = score_mission(write_truth(tmp_path, samples), run)
    assert not report["coverage"]["valid"]
    assert not report["success"]


@pytest.mark.parametrize("problem", ["collision", "missing_contacts", "fall"])
def test_arrival_does_not_hide_missing_or_failed_safety_evidence(tmp_path, problem):
    run = write_run(tmp_path)
    samples = [sample(t / 10) for t in range(1, 25)]
    if problem == "collision":
        samples[1]["contacts"].update(contact_steps=1, contact_count=2,
                                     max_penetration_m=0.004, obstacles=["wB_m"])
    elif problem == "missing_contacts":
        samples[1].pop("contacts")
    else:
        samples[1]["up_alignment"] = 0.1
    report = score_mission(write_truth(tmp_path, samples), run)
    assert report["physical_arrival"]
    assert not report["safety_verified"]
    assert not report["success"]
    if problem == "missing_contacts":
        assert report["safety"]["collision_free"] is None


def test_rejects_mismatched_or_unfinished_mission(tmp_path):
    run = write_run(tmp_path)
    mission = run / "mission.json"
    result = json.loads(mission.read_text())
    result["status"] = "blocked"
    mission.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="differs"):
        mission_interval(run)
    events = run / "events.jsonl"
    events.write_text(events.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="exactly one"):
        mission_interval(run)


def test_planner_false_negative_is_distinct_from_physical_arrival(tmp_path):
    run = write_run(tmp_path, status="blocked")
    report = score_mission(write_truth(tmp_path, [sample(t / 10) for t in range(1, 25)]), run)
    assert report["physical_arrival"]
    assert not report["planner_claimed_arrival"]
    assert not report["success"]
