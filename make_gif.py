"""Record assets/demo.gif from the seed (out of 40) that clears the most pipes."""
import sys
from PIL import Image
from policy import NumpyPolicy, rollout

policy = NumpyPolicy()
best = max(range(40), key=lambda s: rollout(policy, seed=s)[1])
reward, score, steps, frames = rollout(policy, seed=best, record=True, every=2)
imgs = [Image.fromarray(f).resize((216, 384)) for f in frames]
imgs[0].save("assets/demo.gif", save_all=True, append_images=imgs[1:], duration=33, loop=0, optimize=True)
print(f"seed={best} pipes={score} reward={reward:.1f} steps={steps}")
