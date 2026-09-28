"""Torch-free inference for the trained DQN.

The network is just Linear -> ReLU -> Linear, so once the weights are exported
to .npz we can run it with NumPy alone. This keeps the deployed demo tiny
(no PyTorch install needed).
"""
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless pygame

import gymnasium as gym
import numpy as np
import flappy_bird_gymnasium  # noqa: F401  (registers FlappyBird-v0)

WEIGHTS = os.path.join(os.path.dirname(__file__), "runs", "flappybirdv0_weights.npz")


class NumpyPolicy:
    def __init__(self, path=WEIGHTS):
        w = np.load(path)
        self.w1, self.b1 = w["model_0_weight"], w["model_0_bias"]
        self.w2, self.b2 = w["model_2_weight"], w["model_2_bias"]

    def act(self, obs) -> int:
        x = np.asarray(obs, dtype=np.float32)
        h = np.maximum(x @ self.w1.T + self.b1, 0.0)
        return int(np.argmax(h @ self.w2.T + self.b2))


def rollout(policy, seed=None, max_steps=1500, record=False, every=1):
    """Play one episode. Returns (total_reward, pipes_passed, steps, frames)."""
    env = gym.make("FlappyBird-v0", render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    total, score, frames = 0.0, 0, []
    for step in range(max_steps):
        obs, reward, terminated, truncated, info = env.step(policy.act(obs))
        total += reward
        score = info.get("score", score)
        if record and step % every == 0:
            frames.append(env.render())
        if terminated or truncated:
            break
    env.close()
    return total, score, step + 1, frames
