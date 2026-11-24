# Method details

## 1. Simulator (`src/env`)

**World.** 50 m x 50 m arena with walls, ground at z = 0 and a geofence ceiling at 8 m.
Pillars are vertical cylinders (radius 0.3-1.2 m, height 3-14 m); the ones lower than the ceiling can be
over-flown. Moving obstacles are spheres (radius 0.3-0.6 m, 0.5-2 m/s) that change heading at random and
bounce off walls and pillars. Half of them are spawned beside the start-to-goal corridor and fly across it.
Start and goal are at least 20 m apart and kept free of obstacles. `Difficulty` in [0, 1] sets the
obstacle counts (curriculum).

**Flight model.** One decision every 0.2 s, executed as 2 physics steps of 0.1 s. The autopilot tracks
the commanded velocity / yaw rate with first-order lags (0.3 s and 0.15 s) plus a random wind gust:

```
v     <- v     + (R(yaw) [v_fwd 0 v_up]' - v) (1 - exp(-dt / 0.3))
omega <- omega + (omega_cmd - omega)         (1 - exp(-dt / 0.15))
```

**Camera.** For each of the 48 x 48 pixels a ray is intersected analytically with the ground plane, the
walls, every pillar (side and top cap) and every mover. `renderCamera` returns the depth image and a
Lambert-shaded, textured grayscale image with fog, sensor noise and per-episode lighting randomisation.

**Actions.**

| # | action | forward [m/s] | climb [m/s] | yaw rate [rad/s] |
|---|---|---|---|---|
| 1 | forward | 2.5 | 0 | 0 |
| 2 | forward_left | 1.8 | 0 | +0.8 |
| 3 | forward_right | 1.8 | 0 | -0.8 |
| 4 | turn_left | 0.3 | 0 | +1.5 |
| 5 | turn_right | 0.3 | 0 | -1.5 |
| 6 | ascend | 1.0 | +1.2 | 0 |
| 7 | descend | 1.0 | -1.2 | 0 |
| 8 | hover | 0 | 0 | 0 |

**Observation.** `{image, state}`: image = H x W x 3 frame stack in [0, 1]. State = 10 values:
goal distance, sin/cos of the goal bearing in the body frame, height difference to the goal,
altitude, forward/left/up velocity, yaw rate and remaining time. With `useFeatures`, 31 detector
outputs are appended (15 distances, 15 occupancy probabilities, crash risk).

**Reward.**
```
r = 0.5 (d_prev - d)                       progress [m]
  - 0.01                                   time penalty
  - 0.3 max(0, 2 - clearance) / 2          proximity penalty
  - 0.05  if the shield replaced the action
  + 10    goal reached (d < 1.5 m)
  - 10    collision or geofence breach
```

## 2. Automatic labels (`collectData`)

Exploration mixture: 30 % sticky-random, 40 % reactive planner with 25 % noise, 30 % greedy
goal-seeking that ignores obstacles (produces crashes).

| label | source | used by |
|---|---|---|
| sector distance (15) | minimum depth in each cell of the 3 x 5 grid | detector distance head |
| sector occupancy (15) | sector distance < 3 m | detector occupancy head |
| crash risk | crash within the next 5 steps | detector risk head |
| action | executed action | inverse-dynamics head |

## 3. Networks (`src/learning`)

Shared convolutional trunk (`encoderConvLayers`, identical layer names in every network):

| layer | output |
|---|---|
| image input 48 x 48 x 3 | |
| conv 5x5/2, 32 + ReLU | 24 x 24 x 32 |
| conv 3x3/2, 64 + ReLU | 12 x 12 x 64 |
| conv 3x3/2, 64 + ReLU | 6 x 6 x 64 |
| conv 3x3/1, 64 + ReLU | 6 x 6 x 64 |

* **Encoder:** trunk -> FC 256 -> LayerNorm -> tanh.
* **Detector:** trunk -> FC 256 -> FC 128 -> FC 31 (15 occupancy logits, 15 distance logits, 1 risk logit).
  Loss = BCE(occupancy) + 2 SmoothL1(sigmoid(distance) - label) + BCE(risk, positive weight 5).
* **SSL:** CURL logits `z_a' W z_k` between two augmentations (random +-3 px shift, contrast and
  brightness jitter), keys from a momentum encoder (tau = 0.05), InfoNCE loss; inverse dynamics
  `MLP([z_t; z_t+1]) -> action`.
* **Critic:** image -> encoder (initialised from SSL); state -> FC128-FC128; concat -> FC 256 ->
  dueling head `Q = V + A - mean(A)`.

## 4. DQN (`createDQNAgent`, `trainDQN`)

`rlDQNAgent` with `UseDoubleDQN = true`, 3-step returns, discount 0.99, Adam (lr 1e-4, gradient
threshold 10), soft target updates (smoothing factor 0.01), epsilon-greedy from 1.0 to 0.05
(multiplicative decay 2e-5 per step), prioritised replay (alpha 0.6, beta 0.4 -> 1).
Curriculum: difficulty = 0.1 + 0.9 episode / 1200.

## 5. Safety shield (`SafetyShield`)

For every action the shield knows which sectors it flies into (3 rows x 5 columns, row 2 is level
with the camera):

| action | sectors | speed |
|---|---|---|
| forward | middle row, columns 2-4 | 2.5 m/s |
| forward_left / forward_right | middle row, columns 1-3 / 3-5 | 1.8 m/s |
| turn_left / turn_right | middle row, columns 2-3 / 3-4 | 0.3 m/s |
| ascend / descend | top / bottom row columns 2-4 + middle centre | 1.56 / 1.0 m/s |

Rules:
```
distance : min d(sectors) < 0.35 + 0.4 + v * 0.8
TTC      : closing_k = EMA((d_k(t-1) - d_k(t)) / dt);  d_k / closing_k < 1.2 s
risk     : p_crash >= 0.7                      -> no forward / forward_left / forward_right
geofence : altitude > 7.3 m -> no climbing;   altitude < 0.8 m -> no descending
threat   : centre TTC < 1.2 s and closing > own speed + 0.5 m/s -> hovering is unsafe too
fallback : nothing allowed -> keep the action with the largest clearance
```
An unsafe action chosen by the agent is replaced with the reactive planner's best safe action.

## 6. Evaluation

Every policy is run on the same seeded scenarios (`cfg.eval.seedOffset + i`).
SPL = success x L_straight / max(L_path, L_straight).
