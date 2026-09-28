"""Gradio demo: watch the trained DQN agent play Flappy Bird."""
import random
import tempfile

import gradio as gr
from PIL import Image

from policy import NumpyPolicy, rollout

policy = NumpyPolicy()


def play(seed):
    seed = random.randint(0, 10_000) if seed is None else int(seed)
    reward, score, steps, frames = rollout(policy, seed=seed, record=True, every=2)
    imgs = [Image.fromarray(f).resize((216, 384)) for f in frames]
    out = tempfile.NamedTemporaryFile(suffix=".gif", delete=False).name
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=33, loop=0)
    return out, f"Seed {seed}: {score} pipes cleared | total reward {reward:.1f} | {steps} steps"


with gr.Blocks(title="Flappy Bird DQN") as demo:
    gr.Markdown(
        "# 🐦 Flappy Bird — Deep Q-Network\n"
        "A DQN agent (PyTorch) trained from scratch with experience replay and a "
        "target network. Pick a seed or leave it blank for a random one."
    )
    with gr.Row():
        seed = gr.Number(label="Seed (optional)", value=None, precision=0)
        btn = gr.Button("▶ Run agent", variant="primary")
    with gr.Row():
        video = gr.Image(label="Agent playing", type="filepath", height=420)
        info = gr.Textbox(label="Result", interactive=False)
    btn.click(play, seed, [video, info])

if __name__ == "__main__":
    demo.launch()
