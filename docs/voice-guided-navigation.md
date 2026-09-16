# Voice-guided navigation simulator

The `voice-guided-navigation` branch starts at `origin/develop`. The companion
`microduck` branch owns speech input, camera observations, the navigation loop and
bounded movement requests. This repository supplies the same MuJoCo body and
camera used by `robotd --sim`, an apartment scene, and an independent evaluator.

`scene_navigation.xml` includes the existing apartment and all-collisions robot.
It covers the stairwell opening for this initial floor-level navigation task and
adds visible oven, refrigerator, sink and cabinet details to the existing kitchen
furniture. It preserves the walking controller, robot joints, actuator parameters,
doorways and collision furniture. No room-name signs, navigation waypoints or
goal coordinates are sent to the camera planner.

## Run a navigation episode

From this repository on Linux, with the simulator environment installed:

```bash
mkdir -p artifacts/navigation
MUJOCO_GL=egl uv run duck-body \
  --scene src/mjlab_microduck/robot/microduck/scene_navigation.xml \
  --headless --keyframe SIT --cameras a --camera-fps 5 \
  --port 7801 --frame-port 7901 \
  --evaluation-log artifacts/navigation/episode-001.truth.jsonl
```

Connect the companion `robotd` to `127.0.0.1:7801`, and the camera receiver to
`127.0.0.1:7901`. The camera stream is the existing length-prefixed 640×360 UYVY
stream. Like the physical camera, it needs a clockwise 90° display rotation.
`--camera-fps` controls both rendering and transmission rate.
Each camera owns a rendering thread so software rendering cannot pause the
physics loop. A failed camera or a stale rendered frame stops producing bytes,
allowing the navigation receiver's normal frame timeout to stop movement.

The default position is the corridor, outside the kitchen. Optional
`--start-x`, `--start-y` (metres) and `--start-yaw-deg` set a reproducible initial
placement; +90° faces world +y. They apply only before physics starts and the
daemon enables the body. Multiple ducks retain the existing 0.5 m y spacing.
There is no navigation-time teleport/reset action. Select obstacle-free spawn
positions; this placement interface intentionally does not infer a route.
Apartment starts are checked in both the requested pose and HOME before the
daemon can connect. Initial robot geometry must have 20 mm clearance from walls,
furniture and loose objects, and a small support footprint must have flat floor
under it. A failed check explains the obstructing geometry and refuses setup.

## Independently check arrival

After the navigation client says it has arrived and stopped:

```bash
uv run python -m mjlab_microduck.sim.navigation_eval \
  artifacts/navigation/episode-001.truth.jsonl
```

This emits JSON and exits 0 only when the **last** scored samples establish
arrival. Use `--start-time` and `--end-time` in simulator seconds to score exactly
one episode from a longer recording, or `--duck` to select a body. The scorer
also reports whether the duck arrived earlier and subsequently moved away.
An interrupted final JSONL record is ignored; corrupt complete records fail.

For an actual navigation run, bind the score to its exact start/finish events:

```bash
uv run python -m mjlab_microduck.sim.mission_eval \
  artifacts/navigation/episode-001.truth.jsonl \
  ../microduck/apps/navigation/runs/RUN_DIRECTORY
```

This reads `mission.json` and its sibling `events.jsonl`, verifies the recorded
final result matches, and selects only that mission's wall-clock interval. Both
processes must use the same host clock. The report requires continuous truth
coverage with no gap above 0.5 s and retains source hashes for reproduction.
An earlier arrival or a pose reached after the mission ended cannot pass.
The model's arrival claim, physical arrival, final stop acknowledgement and safety
evidence are reported separately; exit 0 requires all four.

New truth logs inspect robot contacts on every physics step (200 Hz), accumulating
wall, furniture, loose-object and other-robot contacts into the 10 Hz records.
Self contacts and named apartment floor supports are excluded. Each window records
the number of contact steps, maximum penetration and obstacle names; brief contacts
between truth records are retained. The first selected window can conservatively
include contact just before mission start. Old logs without contact fields cannot
establish a collision-free run. The mission scorer also checks for released-body
samples with trunk height below 6 cm or tilt above 60° as fall evidence.

The optional truth log is a separate, exclusive-create file; existing evidence
is never overwritten. It records at approximately 10 Hz and is **evaluation
data only**. It must not be sent to the planner or used to choose movement.
The unchanged body sensor protocol's historical `trunk` field is also simulator
truth, and navigation clients must exclude it from planner observations.

The evaluator's apartment-specific success definition is:

- Trunk x in [-3.74, -1.26] and y in [0.76, 2.74] metres, placing the body
  beyond the kitchen doorway with a 20 cm margin from the interior wall faces.
- Trunk height between 0.08 and 0.20 m, tilt at most 30°, linear speed at most
  0.06 m/s, and angular speed at most 0.4 rad/s.
- Torque enabled and the simulator's initial body hold released.
- All of these conditions sustained for at least 1 simulator second, with
  positive sample intervals no greater than 0.5 s.

These bounds belong only to the evaluator. Camera evidence is still required
for the navigation client's own arrival decision. A scorer pass demonstrates
physical arrival in this apartment, not generalization to unseen homes.

## Validation

```bash
uv run --with pytest pytest tests/test_navigation_sim.py tests/test_apartment_flat.py
```

The tests load the real MJCF, verify retained actuators and collision geometry,
check spawn/held-pose orientation, and reject false success from a held start,
fall, doorway-only pose, motion, discontinuous samples or an earlier visit
followed by departure. These checks validate the scene and scoring contract;
an actual camera-and-policy episode is required to establish navigation success.
