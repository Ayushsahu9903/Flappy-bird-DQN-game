"""Gradio demo: watch the trained DQN agent play Flappy Bird."""
import os
import random
import tempfile

import gradio as gr
from PIL import Image

from policy import NumpyPolicy, rollout

policy = NumpyPolicy()


def play():
    seed = random.randint(0, 10_000)
    reward, score, steps, frames = rollout(policy, seed=seed, record=True, every=2)
    imgs = [Image.fromarray(f).resize((216, 384)) for f in frames]
    out = tempfile.NamedTemporaryFile(suffix=".gif", delete=False).name
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=33, loop=0)
    label = "pipe" if score == 1 else "pipes"
    return out, f"{score} {label} cleared | total reward {reward:.1f} | {steps} steps"


with gr.Blocks(title="Flappy Bird DQN") as demo:
    gr.Markdown(
        "# 🐦 Flappy Bird — Deep Q-Network\n"
        "A DQN agent (PyTorch) trained from scratch with experience replay and a "
        "target network. Click the button to watch it play."
    )
    btn = gr.Button("▶ Run agent", variant="primary")
    with gr.Row():
        video = gr.Image(label="Agent playing", type="filepath", height=420)
        info = gr.Textbox(label="Result", interactive=False)
    btn.click(play, None, [video, info])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
