# 🐦 Flappy Bird — Deep Q-Network (PyTorch)

A Deep Q-Learning agent that learns to play Flappy Bird from scratch, built with PyTorch and
[`flappy-bird-gymnasium`](https://github.com/markub3327/flappy-bird-gymnasium).

![demo](assets/demo.gif)

**🎮 Live demo:** https://flappy-bird-dqn-game.onrender.com

> The demo runs on a free Render instance, which sleeps after 15 minutes without visitors.
> If the page is slow to open, give it about a minute to wake up.

## How it works

| Piece | Details |
|---|---|
| Environment | `FlappyBird-v0` (Gymnasium), default 180-dim LiDAR observation, 2 actions (flap / do nothing) |
| Network | MLP: `180 → 256 (ReLU) → 2` ([`dqn.py`](dqn.py)) |
| Experience replay | FIFO buffer of 100k transitions, batch size 32 ([`experience_replay.py`](experience_replay.py)) |
| Target network | Separate copy of the policy net, synced periodically |
| Exploration | ε-greedy, decaying 1.0 → 0.05 (×0.995 per episode) |
| Optimiser / loss | Adam (lr 1e-3), MSE on the Bellman target, γ = 0.99 |

Hyperparameters live in [`parameters.yaml`](parameters.yaml).

## Results

Evaluated greedily (ε = 0) over 100 seeded episodes:

| Metric | Value |
|---|---|
| Mean pipes cleared | 2.4 |
| Best episode | 7 pipes |
| Mean episode reward | 5.6 (max 16.1) |

This is a small learning project, not a state-of-the-art agent. The agent clearly learned to
navigate towards gaps, but it is not robust yet. Ideas for improvement are listed below.

## Run it

```bash
git clone https://github.com/Ayushsahu9903/Flappy-bird-DQN-game.git
cd Flappy-bird-DQN-game
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

**Web demo (no PyTorch needed):**
```bash
pip install -r requirements.txt
python app.py            # opens http://127.0.0.1:7860
```

**Train your own agent:**
```bash
pip install -r requirements-train.txt
python agent.py flappybirdv0 --train      # checkpoints best policy to runs/flappybirdv0.pt
python agent.py flappybirdv0              # watch the saved policy play in a pygame window
```

**Play it yourself:** `python game_flappy_bird.py` (space bar to flap).

If you retrain, re-export the weights for the torch-free demo:
```bash
python export_weights.py runs/flappybirdv0.pt runs/flappybirdv0_weights.npz
```

## Deployment

The demo is deployed on [Render](https://flappy-bird-dqn-game.onrender.com) as a free web service:

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `python app.py`
- **Environment variable:** `PYTHON_VERSION=3.11.9`

The web app uses NumPy-only inference (`policy.py`), so PyTorch is not installed on the server.

## Project layout

```
agent.py                training + evaluation loop
dqn.py                  Q-network
experience_replay.py    replay buffer
parameters.yaml         hyperparameters
policy.py               NumPy-only inference (used by the demo)
app.py                  Gradio web demo
export_weights.py       .pt -> .npz converter
make_gif.py             records assets/demo.gif
game_flappy_bird.py     play the game manually
runs/                   trained checkpoint (.pt), exported weights (.npz), training log
```

## Possible improvements

- Run the gradient update on **every environment step** (currently once per episode) and add a warm-up period
- Double DQN and/or a Huber loss for more stable targets
- Soft target updates (Polyak averaging) instead of hard copies
- Log training curves (TensorBoard / Weights & Biases)
