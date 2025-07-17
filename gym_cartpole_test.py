import os
import gym
import math
import time
import imageio
import numpy as np
import matplotlib.pyplot as plt

# Relative imports
from controller.gym_continuous_cartpole import ContinuousCartPoleEnv
from controller.gym_cartpole_swingup_lqr import swingup_lqr_controller


def run_single_system(masscart, masspole, length, theta_init, thetadot_init, run_idx, save_dir):
    env = ContinuousCartPoleEnv(
        masscart=masscart,
        masspole=masspole,
        length=length,
        render_mode="rgb_array"
    )

    obs, _ = env.reset(options={"init_state": [0.0, 0.0, theta_init, thetadot_init]})
    frames, states, actions = [], [obs], []
    switched = False
    max_steps = 560

    for _ in range(max_steps):
        action, switched = swingup_lqr_controller(obs, switched, masscart, masspole, length)
        obs, reward, done, truncated, _, applied_action = env.step(action)
        frame = env.render()
        frames.append(frame)
        states.append(obs)
        actions.append(applied_action)
        if done or truncated:
            break

    env.close()

    # Final state
    states = np.array(states)
    actions = np.array(actions).squeeze()
    final_theta = states[-1, 2]
    final_theta_dot = states[-1, 3]
    # stabilized = abs((final_theta + np.pi) % (2*np.pi) - np.pi) < 0.2 and abs(final_theta_dot) < 0.5
    theta_wrapped = (final_theta + np.pi) % (2 * np.pi) - np.pi
    stabilized = abs(theta_wrapped) < 0.2 and abs(final_theta_dot) < 0.5
    # Output path and naming
    folder_name = f"mC{masscart:.2f}_mP{masspole:.2f}_L{length:.2f}_th{theta_init:.2f}_dth{thetadot_init:.2f}"
    path = os.path.join(save_dir, f"run_{run_idx:03}_{folder_name}")
    os.makedirs(path, exist_ok=True)

    # Save video
    imageio.mimsave(os.path.join(path, 'cartpole.mp4'), frames, fps=50)

    # Save control plot
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(actions)), actions, label='Control Actions', color='blue', s=10)
    plt.title('Control Actions Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Control Action')
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(path, 'controls.png'))
    plt.close()

    # Save state plots
    plt.figure(figsize=(15, 10))
    labels = ['Cart Position (x)', 'Cart Velocity (x_dot)', 'Pole Angle (theta)', 'Pole Angular Velocity (theta_dot)']
    colors = ['red', 'green', 'orange', 'purple']
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.scatter(range(len(states)), states[:, i], label=labels[i], color=colors[i], s=10)
        plt.title(labels[i] + ' Over Time')
        plt.xlabel('Time Step')
        plt.ylabel(labels[i].split('(')[-1].rstrip(')'))
        plt.grid()
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'states.png'))
    plt.close()

    return {
        "path": path,
        "masscart": masscart,
        "masspole": masspole,
        "length": length,
        "theta_init": theta_init,
        "thetadot_init": thetadot_init,
        "stabilized": stabilized,
        "final_state": states[-1]
    }


if __name__ == "__main__":
    num_runs = 20
    results = []
    save_dir = os.path.join(os.getcwd(), 'videos', 'cartpole_gym_runs')
    os.makedirs(save_dir, exist_ok=True)

    for run_idx in range(num_runs):
        masscart = 2.0
        masspole = np.random.uniform(0.5, 1.0)
        length = np.random.uniform(1.0, 1.5)
        theta_init = np.random.uniform(np.pi - np.pi/2, np.pi + np.pi/2)
        thetadot_init = np.random.uniform(-1.0, 1.0)

        result = run_single_system(masscart, masspole, length, theta_init, thetadot_init, run_idx, save_dir)
        results.append(result)
        print(f"[{run_idx+1}/{num_runs}] Stabilized: {result['stabilized']} — {result['path']}")

    # Summary
    total_stabilized = sum(r['stabilized'] for r in results)
    print(f"\n✅ {total_stabilized} / {num_runs} systems stabilized.")





# import gym
# import time
# import imageio
# import os
# import numpy as np
# import math
# from gym_continuous_cartpole import ContinuousCartPoleEnv
# from gym_cartpole_swingup_lqr import swingup_lqr_controller

# # env = ContinuousCartPoleEnv(render_mode="rgb_array")
# masscart = 2.0
# masspole = np.random.uniform(0.5, 1.0)
# length = np.random.uniform(1.0, 1.5)
# theta_init = np.random.uniform(np.pi-np.pi/2, np.pi+np.pi/2)  # random initial angle in range [-pi/2, pi/2]
# thetadot_init = np.random.uniform(-1.0, 1.0)  # random initial angular velocity
# # env = ContinuousCartPoleEnv(masscart=2.0, masspole=0.5, length=1.0, render_mode="rgb_array")
# env = ContinuousCartPoleEnv(masscart=masscart, masspole=masspole, length=length, render_mode="rgb_array")
# # obs, _ = env.reset()

# # obs, info = env.reset(options={"init_state": [0.0, 0.0, 3*np.pi/4, 0.5]})
# obs, info = env.reset(options={"init_state": [0.0, 0.0, theta_init, thetadot_init]})
# frames = []
# states = []
# actions = []
# states.append(obs)
# switched = False

# for _ in range(1000):
#     # action = np.array([0.0])  # zero control
#     # action = swingup_lqr_controller(obs, env.masscart, env.masspole, env.length)
#     # print(f"obs: {obs}")
#     action, switched = swingup_lqr_controller(obs, switched, env.masscart, env.masspole, env.length)
#     obs, reward, done, truncated, _ , applied_action = env.step(action)
#     frame = env.render()
#     frames.append(frame)
#     states.append(obs)
#     actions.append(applied_action)
#     if done or truncated:
#         obs, _ = env.reset()

# env.close()

# # Save video
# path = os.path.join(os.getcwd(), 'videos', 'cartpole_gym')
# os.makedirs(path, exist_ok=True)

# file_path = os.path.join(path, 'cartpole.mp4')
# # imageio.mimsave(file_path, env.render(mode='rgb_array'), fps=50)
# imageio.mimsave(file_path, frames, fps=50)
# states = np.array(states)
# # print(f"states: {states}")
# # print(f"states.shape: {states.shape}")
# actions = np.array(actions).squeeze()
# # print(f"actions: {actions}")
# # print(f"max action: {np.max(actions)}")
# print(f"cart mass: {env.masscart}")
# print(f"pole mass: {env.masspole}")
# print(f"pole length: {env.length}")
# print(f"final state: {states[-1]}")

# # plot controls vs time
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 5))
# # plt.plot(actions, label='Control Actions')
# plt.scatter(range(len(actions)), actions, label='Control Actions', color='blue', s=10)
# plt.title('Control Actions Over Time')
# plt.xlabel('Time Step')
# plt.ylabel('Control Action')
# plt.legend()
# plt.grid()
# plt.savefig(os.path.join(path, 'controls.png'))
# plt.show()

# # plot states vs time on separate subplots
# plt.figure(figsize=(15, 10))
# plt.subplot(2, 2, 1)
# plt.scatter(range(len(states)), states[:, 0], label='Cart Position (x)', color='red', s=10)
# plt.title('Cart Position Over Time')
# plt.xlabel('Time Step')
# plt.ylabel('Position (x)')
# plt.grid() 
# plt.legend()

# plt.subplot(2, 2, 2)
# plt.scatter(range(len(states)), states[:, 1], label='Cart Velocity (x_dot)', color='green', s=10)
# plt.title('Cart Velocity Over Time')
# plt.xlabel('Time Step')
# plt.ylabel('Velocity (x_dot)')
# plt.grid()
# plt.legend()

# plt.subplot(2, 2, 3)
# plt.scatter(range(len(states)), states[:, 2], label='Pole Angle (theta)', color='orange', s=10)
# plt.title('Pole Angle Over Time')
# plt.xlabel('Time Step')
# plt.ylabel('Angle (theta)')
# plt.grid()
# plt.legend()

# plt.subplot(2, 2, 4)
# plt.scatter(range(len(states)), states[:, 3], label='Pole Angular Velocity (theta_dot)', color='purple', s=10)
# plt.title('Pole Angular Velocity Over Time')
# plt.xlabel('Time Step')
# plt.ylabel('Angular Velocity (theta_dot)')
# plt.grid()
# plt.legend()

# plt.tight_layout()
# plt.savefig(os.path.join(path, 'states.png'))
# plt.show()

