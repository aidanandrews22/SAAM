from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict

import imageio.v2 as imageio  # type: ignore – needed for video export
import matplotlib.pyplot as plt  # plots are written to disk, never shown
import numpy as np
import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from controller.gym_cartpole_swingup_lqr import swingup_lqr_controller
from generate_dataset_cartpole_gym import (
    run_single_trajectory,
    get_valid_masses_and_lengths_uniform,
    generate_random_initial_conditions,
)
from model.models import build_model
from quinine import QuinineArgumentParser
from model.schema import schema as quinine_schema

GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_SEED)


def generate_ground_truth(
    cartmass: float,
    polemass: float,
    polelength: float,
    n_points: int,
    dt: float = 0.025,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    init_state = generate_random_initial_conditions(size=1)[0]

    states, controls, modes = run_single_trajectory(
        cartmass, polemass, polelength, init_state, n_points, dt
    )
    return states, controls, modes


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean‑squared error helper handling shape alignment."""
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
    return float(np.mean((a - b) ** 2))


def build_and_load_model(ckpt_path: Path, device: torch.device):
    """Instantiate model from training config and load weights."""

    parser = QuinineArgumentParser(schema=quinine_schema)
    args = parser.parse_quinfig()

    model = build_model(args.model).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model, args


def predict_sequence(
    model: torch.nn.Module,
    gt_states: np.ndarray,
    gt_controls: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Feed ground‑truth state/control history and obtain model predictions.

    The model receives the *full* trajectory tensors at once; adjust here if
    your model expects recurrent rollout.
    """
    control_modes = np.zeros_like(gt_controls)
    ctrl_in = np.stack([gt_controls, control_modes], axis=-1)  # (T-1, 2)

    with torch.no_grad():
        states_t = (
            torch.from_numpy(gt_states[:-1]).float().unsqueeze(0).to(device)
        )  # (1, T-1, 4)
        ctrl_t = (
            torch.from_numpy(ctrl_in).float().unsqueeze(0).to(device)
        )  # (1, T-1, 2)
        pred_ctrl, pred_states = model(
            states_t, ctrl_t
        )  # outputs are (1, T-1, 1) and (1, T-1, 4)

    return (
        pred_ctrl.squeeze(0).cpu().numpy(),  # (T-1, 1)
        pred_states.squeeze(0).cpu().numpy(),  # (T-1, 4)
    )


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------


def plot_scatter(
    x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str, path: Path
):
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, alpha=0.7, edgecolors="k")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_hist(data: np.ndarray, xlabel: str, title: str, path: Path):
    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=30, edgecolor="k", alpha=0.8)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def evaluate(
    ckpt_path: Path,
    num_runs: int = 100,
    n_points: int = 300,
    dt: float = 0.025,
    save_video_first_n: int = 0,
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, _ = build_and_load_model(ckpt_path, device)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("eval_results") / f"{ts}_runs{num_runs}"
    plots_dir = out_dir / "plots"
    videos_dir = out_dir / "videos"
    plots_dir.mkdir(parents=True, exist_ok=True)
    if save_video_first_n > 0:
        videos_dir.mkdir(parents=True, exist_ok=True)

    ctrl_mse, state_mse = [], []
    pole_masses, pole_lengths = [], []
    ctrl_ranges = []  # additional statistics

    rng = np.random.default_rng(GLOBAL_SEED)

    for run in tqdm(range(num_runs), desc="evaluating"):
        cartmass = 2.0  # fixed, as in training
        polemass, polelength = get_valid_masses_and_lengths_uniform(size=1)[1:]
        polemass = polemass.item()
        polelength = polelength.item()

        gt_states, gt_controls, modes = generate_ground_truth(
            cartmass, polemass, polelength, n_points, dt
        )

        pred_controls, pred_states = predict_sequence(
            model, gt_states, gt_controls, device
        )
        pred_controls = pred_controls[:, 0]  # drop singleton dim

        ctrl_mse.append(mse(gt_controls, pred_controls))
        state_mse.append(mse(gt_states[1:], pred_states))

        pole_masses.append(polemass)
        pole_lengths.append(polelength)
        ctrl_ranges.append(gt_controls.max() - gt_controls.min())

        if run < save_video_first_n:
            from controller.gym_continuous_cartpole import ContinuousCartPoleEnv

            env = ContinuousCartPoleEnv(
                masscart=cartmass,
                masspole=polemass,
                length=polelength,
                render_mode="rgb_array",
            )
            env.tau = dt
            env.screen_width, env.screen_height = 1200, 600
            obs, _ = env.reset(options={"init_state": gt_states[0].tolist()})
            frames: List[np.ndarray] = []
            switched = bool(modes[0])  # initial mode flag, negligible
            for u in gt_controls:
                action, switched = swingup_lqr_controller(
                    obs, switched, cartmass, polemass, polelength
                )
                obs, *_ = env.step(action)
                frames.append(env.render())
            env.close()
            video_path = videos_dir / f"run_{run:03d}.mp4"
            imageio.mimsave(video_path, frames, fps=int(1 / dt))

    ctrl_mse_arr = np.array(ctrl_mse)
    state_mse_arr = np.array(state_mse)

    summary = {
        "avg_control_mse": float(ctrl_mse_arr.mean()),
        "std_control_mse": float(ctrl_mse_arr.std()),
        "avg_state_mse": float(state_mse_arr.mean()),
        "std_state_mse": float(state_mse_arr.std()),
    }

    metrics_file = out_dir / "metrics.txt"
    with metrics_file.open("w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v:.6f}\n")

    pole_masses = np.array(pole_masses)
    pole_lengths = np.array(pole_lengths)

    plot_scatter(
        pole_masses,
        ctrl_mse_arr,
        xlabel="pole mass (kg)",
        ylabel="control MSE",
        title="Control MSE vs pole mass",
        path=plots_dir / "ctrl_mse_vs_polemass.png",
    )
    plot_scatter(
        pole_lengths,
        ctrl_mse_arr,
        xlabel="pole length (m)",
        ylabel="control MSE",
        title="Control MSE vs pole length",
        path=plots_dir / "ctrl_mse_vs_polelen.png",
    )

    plot_scatter(
        pole_masses,
        state_mse_arr,
        xlabel="pole mass (kg)",
        ylabel="state MSE",
        title="State MSE vs pole mass",
        path=plots_dir / "state_mse_vs_polemass.png",
    )
    plot_scatter(
        pole_lengths,
        state_mse_arr,
        xlabel="pole length (m)",
        ylabel="state MSE",
        title="State MSE vs pole length",
        path=plots_dir / "state_mse_vs_polelen.png",
    )

    plot_hist(
        ctrl_mse_arr,
        xlabel="control MSE",
        title="Distribution of control MSE across runs",
        path=plots_dir / "hist_control_mse.png",
    )
    plot_hist(
        state_mse_arr,
        xlabel="state MSE",
        title="Distribution of state MSE across runs",
        path=plots_dir / "hist_state_mse.png",
    )

    print("\nEvaluation complete – summary:")
    for k, v in summary.items():
        print(f"  {k}: {v:.6f}")
    print(f"\nOutputs written to: {out_dir}")

    return summary


def main():
    start = time.time()
    evaluate(
        ckpt_path=Path(
            "output_gym_full/fb217d29-d37e-421e-bc59-20c49c439a87/checkpoint_epoch100_step250000.pt"
        ),
        num_runs=100,
        n_points=560,
        save_video_first_n=3,
    )
    dur = time.time() - start
    print(f"Total wall‑clock time: {dur / 60:.1f} min")


if __name__ == "__main__":
    main()
