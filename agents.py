from collections import defaultdict
import numpy as np
import random

class QLearningAgent:

    def __init__(self, alpha, epsilon, discount, n_actions):
        self.n_actions = n_actions
        self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount
        self.episode_rewards = []
        self.episode_lengths = []

    def get_qvalue(self, state, action):
        return self._qvalues[state][action]

    def set_qvalue(self,state,action,value):
        self._qvalues[state][action] = value

    def get_value(self, state):
        possible_actions = range(self.n_actions)

        if len(possible_actions) == 0:
            return 0.0

        action_values = []
        for a in possible_actions:
            action_values.append(self.get_qvalue(state, a))
        state_value = np.max(action_values)

        return state_value

    def update(self, state, action, reward, next_state):
        gamma = self.discount
        learning_rate = self.alpha

        Q = self.get_qvalue(state, action)
        Q = (1 - self.alpha) * Q + self.alpha * (reward + gamma * self.get_value(next_state))
        self.set_qvalue(state, action, Q)

    def get_best_action(self, state):
        possible_actions = range(self.n_actions)

        if len(possible_actions) == 0:
            return None

        action_values = []
        for a in possible_actions:
            action_values.append(self.get_qvalue(state, a))
        best_action = np.argmax(action_values)

        return best_action

    def get_action(self, state):
        possible_actions = range(self.n_actions)
        action = None

        if len(possible_actions) == 0:
            return None

        epsilon = self.epsilon

        probs = np.ones(self.n_actions, dtype=float) * self.epsilon / self.n_actions
        probs[self.get_best_action(state)] += (1.0 - self.epsilon)
        chosen_action = np.random.choice(possible_actions, p=probs)

        return chosen_action

    def train(self, env, n_episodes=5000, t_max=10**4, verbose=True, verbose_per_episode=100):
        print("\nTraining {}.".format(self.__class__.__name__))
        for i in range(n_episodes):
            episode_reward = 0.0
            s = env.reset()
            for t in range(t_max):
                a = self.get_action(s)
                next_s,r,done,_ = env.step(a)

                self.update(s, a, r, next_s)
                s = next_s
                episode_reward +=r
                if done:
                    self.episode_lengths.append(t+1)
                    break
            self.episode_rewards.append(episode_reward)

            if verbose:
                if i % verbose_per_episode == 0:
                    print('Episode {} done. Mean reward = {}'.format(i+1, np.mean(self.episode_rewards[-100:])))
        print('Training {} episodes done. Mean reward = {}\n'.format(n_episodes, np.mean(self.episode_rewards[-100:])))

class EVSarsaAgent(QLearningAgent):

    def get_value(self, state):
        epsilon = self.epsilon
        possible_actions = range(self.n_actions)

        if len(possible_actions) == 0:
            return 0.0

        probs = np.ones(self.n_actions, dtype=float) * self.epsilon / self.n_actions
        probs[self.get_best_action(state)] += (1.0 - self.epsilon)
        state_value = 0
        for a in possible_actions:
            state_value += probs[a] * self.get_qvalue(state, a)

        return state_value