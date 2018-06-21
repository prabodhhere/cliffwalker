import numpy as np
from matplotlib import pyplot as plt

def draw_policy(env, agent):
    n_rows, n_cols = env._cliff.shape
    actions = '^>v<'

    for yi in range(n_rows):
        for xi in range(n_cols):
            if env._cliff[yi, xi]:
                print(" C ", end='')
            elif (yi * n_cols + xi) == env.start_state_index:
                print(" X ", end='')
            elif (yi * n_cols + xi) in [np.ravel_multi_index(env.end[i], env.shape) for i in range(len(env.end))]:
                print(" T ", end='')
            else:
                print(" %s " % actions[agent.get_best_action(yi * n_cols + xi)], end='')
        print()

def plot_episode_stats(agent):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(agent.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    plt.show(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10,5))
    # rewards_smoothed = pd.Series(agent.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(agent.episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time")
    plt.show(fig2)

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10,5))
    plt.plot(np.cumsum(agent.episode_lengths), np.arange(len(agent.episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    plt.show(fig3)

def plot_stats(agent_ql, agent_sarsa):
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(agent_ql.episode_lengths, label='Q Learning')
    plt.plot(agent_sarsa.episode_lengths, label='SARSA')
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    plt.legend()
    plt.show(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10,5))
    # rewards_smoothed = pd.Series(agent_ql.episode_rewards).rolling(10, min_periods=smoothing_window).mean()
    plt.plot(agent_ql.episode_rewards, label='Q Learning')
    plt.plot(agent_sarsa.episode_rewards, label='SARSA')
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("Episode Reward over Time")
    plt.legend()
    plt.show(fig2)

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10,5))
    plt.plot(np.cumsum(agent_ql.episode_lengths), np.arange(len(agent_ql.episode_lengths)), label='Q Learning')
    plt.plot(np.cumsum(agent_sarsa.episode_lengths), np.arange(len(agent_sarsa.episode_lengths)), label='SARSA')
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    plt.legend()
    plt.show(fig3)