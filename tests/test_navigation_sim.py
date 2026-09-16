"""Navigation scene and independent scoring; no camera/planner outcome is faked here."""

import json
import math
import threading
import time

import mujoco
import numpy as np
import pytest

from mjlab_microduck.sim.body_server import Body, HOME_TRUNK_Z, SCENES, World
from mjlab_microduck.sim.navigation_eval import ArrivalEvaluator, SCHEMA, TruthLogger, score_log


@pytest.fixture(scope="module")
def world():
    return World(SCENES / "scene_navigation.xml")


def test_scene_preserves_robot_actuators_and_all_collision_geometry(world):
    baseline = mujoco.MjModel.from_xml_path(str(SCENES / "scene_apartment_flat.xml"))
    actual = world.model
    assert actual.nu == baseline.nu == 14
    np.testing.assert_array_equal(actual.actuator_gainprm, baseline.actuator_gainprm)
    np.testing.assert_array_equal(actual.actuator_biasprm, baseline.actuator_biasprm)
    for geom in range(baseline.ngeom):
        if not baseline.geom_contype[geom] and not baseline.geom_conaffinity[geom]:
            continue
        name = mujoco.mj_id2name(baseline, mujoco.mjtObj.mjOBJ_GEOM, geom)
        if name:
            candidate = mujoco.mj_name2id(actual, mujoco.mjtObj.mjOBJ_GEOM, name)
        else:
            # MuJoCo inserts new world geoms before robot geoms. Match unnamed
            # robot shapes within their body, never by a global geom index.
            body = baseline.geom_bodyid[geom]
            body_name = mujoco.mj_id2name(baseline, mujoco.mjtObj.mjOBJ_BODY, body)
            actual_body = mujoco.mj_name2id(actual, mujoco.mjtObj.mjOBJ_BODY, body_name)
            candidate = actual.body_geomadr[actual_body] + geom - baseline.body_geomadr[body]
        np.testing.assert_array_equal(actual.geom_size[candidate], baseline.geom_size[geom])
        np.testing.assert_array_equal(actual.geom_pos[candidate], baseline.geom_pos[geom])
    assert actual.geom("nav_oven_glass").contype == 0
    assert actual.geom("nav_fridge_handle_top").contype == 0


def test_placement_and_held_pose_restore_preserve_yaw(world):
    body = Body(world, 0)
    body.place(None, HOME_TRUNK_Z, offset_y=-1.4, offset_x=-0.25, yaw=math.pi / 2)
    expected = np.array([-0.25, -1.4, HOME_TRUNK_Z, math.sqrt(0.5), 0, 0, math.sqrt(0.5)])
    np.testing.assert_allclose(world.data.qpos[body.trunk:body.trunk + 7], expected)
    world.data.qpos[body.trunk] = 9
    body.restore()
    np.testing.assert_allclose(world.data.qpos[body.trunk:body.trunk + 7], expected)
    with pytest.raises(ValueError, match="finite"):
        body.place(None, HOME_TRUNK_Z, 0, offset_x=math.nan)


def sample(t, **kwargs):
    result = {
        "type": "sample", "duck": 0, "sim_time": t,
        "position_m": [-1.5, 2.0, 0.125], "up_alignment": 1.0,
        "speed_mps": 0.0, "angular_speed_rps": 0.0, "released": True, "torque_on": True,
    }
    result.update(kwargs)
    return result


def test_arrival_requires_full_body_entry_upright_stopped_and_a_dwell():
    evaluator = ArrivalEvaluator()
    for step in range(10):
        assert not evaluator.update(sample(step / 10))["arrived"]
    assert evaluator.update(sample(1.0))["arrived"]
    assert not evaluator.update(sample(1.1, position_m=[-0.8, 2.0, 0.125]))["arrived"]
    assert evaluator.first_arrival == 1.0


@pytest.mark.parametrize("invalid", [
    {"position_m": [-1.1, 2.0, 0.125]},  # doorway, not inside
    {"position_m": [-1.5, 2.0, 0.05]},  # fallen with trunk still upright
    {"position_m": [-1.5, 2.0, 0.50]},  # on top of furniture
    {"up_alignment": 0.5},
    {"speed_mps": 0.2},
    {"angular_speed_rps": 1.0},
    {"released": False},  # simulator's held start is not policy success
    {"torque_on": False},
    {"speed_mps": math.nan},
])
def test_invalid_arrival_never_passes(invalid):
    evaluator = ArrivalEvaluator()
    for step in range(25):
        assert not evaluator.update(sample(step / 10, **invalid))["arrived"]


def test_missing_or_repeated_samples_do_not_count_as_stopped_dwell():
    evaluator = ArrivalEvaluator()
    evaluator.update(sample(0))
    assert not evaluator.update(sample(5))["arrived"]
    for _ in range(20):
        assert not evaluator.update(sample(5))["arrived"]


def test_truth_log_records_held_start_but_cannot_mark_it_success(tmp_path, world):
    body = Body(world, 0)
    body.place(None, HOME_TRUNK_Z, offset_y=2.0, offset_x=-1.5)
    world.bodies = [body]
    path = tmp_path / "truth.jsonl"
    logger = TruthLogger(path, SCENES / "scene_navigation.xml", world)
    for _ in range(70):
        world.step(4)
        logger.sample()
    logger.close()
    report = score_log(path)
    assert report["samples"] >= 10
    assert report["final_conditions"]["inside_kitchen"]
    assert not report["final_conditions"]["policy_support"]
    assert not report["success"]
    assert not report["ever_arrived"]
    with pytest.raises(FileExistsError):
        TruthLogger(path, SCENES / "scene_navigation.xml", world)


def test_score_uses_final_state_and_filters_time_range(tmp_path):
    path = tmp_path / "truth.jsonl"
    records = [{"type": "metadata", "schema": SCHEMA, "scene": "scene_navigation.xml"}]
    records += [sample(step / 10) for step in range(13)]
    records.append(sample(1.3, position_m=[0, 0, 0.125]))
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    assert score_log(path, end_time=1.2)["success"]
    report = score_log(path)
    assert report["ever_arrived"]
    assert not report["success"]
    assert not score_log(path, start_time=0.5, end_time=1.2)["success"]
    with pytest.raises(ValueError, match="no samples"):
        score_log(path, duck=1)


def test_slow_camera_does_not_hold_physics_and_closes_gl_on_its_thread(monkeypatch, world):
    from mjlab_microduck.sim import camera

    rendering = threading.Event()
    release = threading.Event()
    owner_threads = []

    class SlowCamera:
        def __init__(self, model, name):
            owner_threads.append(threading.get_ident())
            self.renderer = self

        def render(self, world):
            owner_threads.append(threading.get_ident())
            rendering.set()
            assert release.wait(timeout=5)

        def frame(self, max_age):
            return b"frame"

        def close(self):
            owner_threads.append(threading.get_ident())

    monkeypatch.setattr(camera, "Camera", SlowCamera)
    worker = camera.CameraWorker(world, "head_camera", fps=1)
    try:
        assert rendering.wait(timeout=2)
        before = world.data.time
        started = time.monotonic()
        world.step(4)
        assert world.data.time > before
        assert time.monotonic() - started < 0.5
        assert worker.frame() == b"frame"
    finally:
        release.set()
        worker.close()
    assert not worker.thread.is_alive()
    assert len(set(owner_threads)) == 1
    assert owner_threads[0] != threading.get_ident()
    assert worker.frame() is None


def test_camera_does_not_repeat_stale_frames_forever(monkeypatch):
    from mjlab_microduck.sim.camera import Camera

    camera = Camera.__new__(Camera)
    camera.lock = threading.Lock()
    camera.latest = b"frame"
    camera.latest_at = time.monotonic() - 4
    assert camera.frame(max_age=2) is None
    assert camera.frame(max_age=5) == b"frame"
