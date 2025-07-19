import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import imageio
from datetime import datetime
from tqdm import tqdm

from controller.gym_continuous_cartpole import ContinuousCartPoleEnv
from controller.gym_cartpole_swingup_lqr import swingup_lqr_controller
from model.models import build_model


def generate_test_trajectory(
    cartmass, polemass, polelength, n_points, initial_state, dt=0.025, record_video=False
):
    """Generate a test trajectory using the controller."""
    render_mode = "rgb_array" if record_video else None
    env = ContinuousCartPoleEnv(
        masscart=cartmass,
        masspole=polemass,
        length=polelength,
        render_mode=render_mode,
    )
    env.tau = dt
    
    # Set wider screen for better visibility
    if record_video:
        env.screen_width = 1200
        env.screen_height = 600

    obs, _ = env.reset(options={"init_state": initial_state.tolist()})

    states = [obs]
    controls = []
    frames = []
    switched = False

    for _ in range(n_points - 1):
        action, switched = swingup_lqr_controller(
            obs, switched, cartmass, polemass, polelength
        )

        obs, _, done, truncated, _, applied_action = env.step(action)
        
        if record_video:
            frame = env.render()
            frames.append(frame)

        states.append(obs)
        controls.append(applied_action[0])

        if done or truncated:
            while len(states) < n_points:
                states.append(obs)
            while len(controls) < n_points - 1:
                controls.append(applied_action[0])
            break

    env.close()

    states = np.array(states)
    controls = np.array(controls)

    return states, controls, frames if record_video else None


def generate_model_trajectory(
    model, cartmass, polemass, polelength, n_points, initial_state, device="cuda:0", dt=0.025
):
    """Generate a trajectory using the model predictions."""
    env = ContinuousCartPoleEnv(
        masscart=cartmass,
        masspole=polemass,
        length=polelength,
        render_mode="rgb_array",
    )
    env.tau = dt
    
    # Set wider screen for better visibility
    env.screen_width = 1200
    env.screen_height = 600

    obs, _ = env.reset(options={"init_state": initial_state.tolist()})

    states = [obs]
    controls = []
    frames = []
    
    for _ in range(n_points - 1):
        # Get model prediction for next control action
        state_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        
        # Create dummy control input for the model
        dummy_control = np.array([[0.0, 0.0]])  # [control_value, control_mode]
        control_tensor = torch.tensor(dummy_control, dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            pred_controls, _ = model(state_tensor, control_tensor)
            action = pred_controls[0, 0, 0].cpu().numpy()

        obs, _, done, truncated, _, applied_action = env.step(action)
        frame = env.render()
        frames.append(frame)

        states.append(obs)
        controls.append(applied_action[0])

        if done or truncated:
            while len(states) < n_points:
                states.append(obs)
            while len(controls) < n_points - 1:
                controls.append(applied_action[0])
            break

    env.close()

    states = np.array(states)
    controls = np.array(controls)

    return states, controls, frames


def evaluate_model_predictions(model, states, controls, device="cuda:0"):
    """Get model predictions for given states and controls."""
    model.eval()

    states_tensor = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(
        0
    )
    controls_tensor = torch.tensor(
        controls, dtype=torch.float32, device=device
    ).unsqueeze(0)

    # Create control labels (zeros for simplicity since we only care about control values)
    control_modes = np.zeros_like(controls)
    control_input = np.stack([controls, control_modes], axis=-1)
    control_input_tensor = torch.tensor(
        control_input, dtype=torch.float32, device=device
    ).unsqueeze(0)

    with torch.no_grad():
        pred_controls, pred_states = model(states_tensor, control_input_tensor)

    return pred_controls.squeeze().cpu().numpy(), pred_states.squeeze().cpu().numpy()


def compute_mse(true_values, predicted_values):
    """Compute MSE between true and predicted values."""
    return np.mean((true_values - predicted_values) ** 2)


def check_stabilization(states):
    """Check if the system is stabilized (pole upright)."""
    final_state = states[-1]
    final_theta = final_state[2]  # theta (pole angle)
    final_theta_dot = final_state[3]  # theta_dot (angular velocity)
    
    # Wrap theta to [-pi, pi]
    theta_wrapped = (final_theta + np.pi) % (2 * np.pi) - np.pi
    stabilized = abs(theta_wrapped) < 0.2 and abs(final_theta_dot) < 0.5
    
    return stabilized


def plot_control_comparison(true_controls, pred_controls, run_idx, save_dir):
    """Plot control actions comparison."""
    plt.figure(figsize=(12, 6))
    time_steps = range(len(true_controls))
    
    plt.plot(time_steps, true_controls, 'b-', label='Controller (Ground Truth)', linewidth=2)
    plt.plot(time_steps, pred_controls, 'r--', label='Model Prediction', linewidth=2, alpha=0.8)
    
    plt.title(f'Control Actions Comparison - Run {run_idx}')
    plt.xlabel('Time Step')
    plt.ylabel('Control Action')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'control_comparison_run_{run_idx}.png'), dpi=150)
    plt.close()


def compute_control_stats(controls, label):
    """Compute statistics for control actions."""
    return {
        f'{label}_max': np.max(controls),
        f'{label}_min': np.min(controls),
        f'{label}_mean': np.mean(controls),
        f'{label}_std': np.std(controls),
        f'{label}_range': np.max(controls) - np.min(controls)
    }


def run_evaluation(model_path, num_runs, device="cuda:0"):
    """Run evaluation comparing controller vs model."""

    # Load model
    from quinine import QuinineArgumentParser
    from model.schema import schema

    # Use the original config file
    config_path = "conf/aidan_cartpole.yaml"
    if os.path.exists(config_path):
        parser = QuinineArgumentParser(schema=schema)
        args = parser.parse_quinfig()
    else:
        raise FileNotFoundError(f"Config file not found at {config_path}")

    model = build_model(args.model)
    model = model.to(device)

    # Load model weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"eval_results/{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Create subdirectories
    videos_dir = os.path.join(save_dir, "videos")
    plots_dir = os.path.join(save_dir, "plots")
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    total_control_mse = 0.0
    total_state_mse = 0.0
    controller_stabilized = 0
    model_stabilized = 0
    
    # Aggregate statistics
    all_controller_stats = []
    all_model_stats = []
    
    # Save first few runs for plotting and videos
    plot_runs = min(5, num_runs)

    for run in tqdm(range(num_runs), desc="Evaluating"):
        # Generate random parameters
        cartmass = 2.0
        polemass = np.random.uniform(0.3, 1.0)
        polelength = np.random.uniform(1.0, 1.5)

        # Generate random initial state
        initial_state = np.array(
            [
                0.0,  # x position
                np.random.uniform(np.pi - np.pi / 2, np.pi + np.pi / 2),  # theta
                0.0,  # x velocity
                np.random.uniform(-1.0, 1.0),  # theta velocity
            ]
        )

        n_points = 300  # Match training data

        # Generate controller trajectory (with video for first few runs)
        record_video = run < plot_runs
        states, controls, controller_frames = generate_test_trajectory(
            cartmass, polemass, polelength, n_points, initial_state, record_video=record_video
        )

        # Get model predictions
        pred_controls, pred_states = evaluate_model_predictions(
            model, states, controls, device
        )

        # Generate model trajectory with video for first few runs
        if record_video:
            model_states, model_controls, model_frames = generate_model_trajectory(
                model, cartmass, polemass, polelength, n_points, initial_state, device
            )
            
            # Save videos
            if controller_frames:
                controller_video_path = os.path.join(videos_dir, f'controller_run_{run}.mp4')
                imageio.mimsave(controller_video_path, controller_frames, fps=40)
            
            model_video_path = os.path.join(videos_dir, f'model_run_{run}.mp4')
            imageio.mimsave(model_video_path, model_frames, fps=40)

        # Check stabilization
        controller_stable = check_stabilization(states)
        model_stable = check_stabilization(pred_states)
        
        if controller_stable:
            controller_stabilized += 1
        if model_stable:
            model_stabilized += 1

        # Compute control statistics
        controller_stats = compute_control_stats(controls, 'controller')
        model_controls_trimmed = pred_controls[:-1, 0]  # Remove last prediction
        model_stats = compute_control_stats(model_controls_trimmed, 'model')
        
        all_controller_stats.append(controller_stats)
        all_model_stats.append(model_stats)

        # Plot first few runs
        if run < plot_runs:
            plot_control_comparison(controls, model_controls_trimmed, run, plots_dir)

        # Compute MSE for controls (compare predicted vs actual)
        control_mse = compute_mse(
            controls, model_controls_trimmed
        )

        # Compute MSE for states (compare predicted next states vs actual next states)
        state_mse = compute_mse(states[1:], pred_states[:-1])  # Compare next states

        total_control_mse += control_mse
        total_state_mse += state_mse

    # Compute averages
    avg_control_mse = total_control_mse / num_runs
    avg_state_mse = total_state_mse / num_runs
    
    # Aggregate control statistics
    controller_max_vals = [stats['controller_max'] for stats in all_controller_stats]
    controller_min_vals = [stats['controller_min'] for stats in all_controller_stats]
    model_max_vals = [stats['model_max'] for stats in all_model_stats]
    model_min_vals = [stats['model_min'] for stats in all_model_stats]

    # Print results
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS ({num_runs} runs)")
    print(f"{'='*60}")
    
    print(f"\nMSE Results:")
    print(f"  Average Control MSE: {avg_control_mse:.6f}")
    print(f"  Average State MSE: {avg_state_mse:.6f}")
    print(f"  Total MSE: {avg_control_mse + avg_state_mse:.6f}")
    
    print(f"\nStabilization Results:")
    print(f"  Controller stabilized: {controller_stabilized}/{num_runs} ({100*controller_stabilized/num_runs:.1f}%)")
    print(f"  Model stabilized: {model_stabilized}/{num_runs} ({100*model_stabilized/num_runs:.1f}%)")
    
    print(f"\nControl Action Statistics:")
    print(f"  Controller - Max: {np.mean(controller_max_vals):.3f} ± {np.std(controller_max_vals):.3f}")
    print(f"  Controller - Min: {np.mean(controller_min_vals):.3f} ± {np.std(controller_min_vals):.3f}")
    print(f"  Model - Max: {np.mean(model_max_vals):.3f} ± {np.std(model_max_vals):.3f}")
    print(f"  Model - Min: {np.mean(model_min_vals):.3f} ± {np.std(model_min_vals):.3f}")
    
    print(f"\nOutputs saved to: {save_dir}/")
    print(f"Generated {plot_runs} control comparison plots and videos")
    print(f"  - Videos: {videos_dir}/")
    print(f"  - Plots: {plots_dir}/")

    return avg_control_mse, avg_state_mse


def main():
    # Hard-coded values
    model_path = "output_gym_full/fb217d29-d37e-421e-bc59-20c49c439a87/checkpoint_epoch100_step250000.pt"
    num_runs = 100
    device = "cuda:0"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    run_evaluation(model_path, num_runs, device)


if __name__ == "__main__":
    main()
