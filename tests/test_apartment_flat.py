"""The beginner navigation scene must not retain the apartment's unguarded stairwell."""

from pathlib import Path

import mujoco
import numpy as np
import pytest


SCENES = Path(__file__).resolve().parents[1] / "src/mjlab_microduck/robot/microduck"


def floor_height(model, data, x, y):
    geom_id = np.array([-1], dtype=np.int32)
    distance = mujoco.mj_ray(
        model, data, np.array([x, y, 0.5]), np.array([0.0, 0.0, -1.0]),
        None, True, -1, geom_id,
    )
    return 0.5 - distance


def test_flat_scene_covers_every_step_without_changing_original():
    flat = mujoco.MjModel.from_xml_path(str(SCENES / "scene_apartment_flat.xml"))
    original = mujoco.MjModel.from_xml_path(str(SCENES / "scene_apartment.xml"))
    flat_data, original_data = mujoco.MjData(flat), mujoco.MjData(original)
    mujoco.mj_forward(flat, flat_data)
    mujoco.mj_forward(original, original_data)
    for x in (-0.39, -0.20, -0.01):
        for y in (-1.39, -1.3125, -1.1375, -0.9625, -0.7875, -0.71):
            assert floor_height(flat, flat_data, x, y) == pytest.approx(0.0)
            assert floor_height(original, original_data, x, y) < -0.09
    assert flat.ngeom == original.ngeom + 1
