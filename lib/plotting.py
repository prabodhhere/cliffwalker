import numpy as np

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