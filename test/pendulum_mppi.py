import gym
import numpy as np

from base.base_mpc import MPPI

ENV_NAME = "Pendulum-v0"

TIMESTEPS = 50  # T
N_SAMPLES = 1000  # K
ACTION_LOW = -2.0
ACTION_HIGH = 2.0

noise_mu = 0
noise_sigma = 1
lambda_ = 1


def pendulum_MPPI():
    """ MMPI according to algorithm 2 in Williams et al., 2017
    'Information Theoretic MPC for Model-Based Reinforcement Learning' """

    env = gym.make(ENV_NAME)

    def _ensure_non_zero(costs, beta, factor):
        return np.exp(-factor*(costs - beta))

    env.reset()
    state = env.env.state

    while True:
        U = np.random.uniform(low=ACTION_LOW, high=ACTION_HIGH, size=TIMESTEPS)  # pendulum joint effort in (-2, +2)
        cost_total = np.zeros(shape=(N_SAMPLES))
        noise = np.random.normal(loc=noise_mu, scale=noise_sigma, size=(N_SAMPLES, TIMESTEPS))

        for k in range(N_SAMPLES):
            env.env.state = state
            for t in range(1,TIMESTEPS):
                perturbed_action_t = U[t-1] + noise[k, t-1]
                state_next, reward, _, _ = env.step([perturbed_action_t])
                cost_total[k] += -reward

        beta = np.min(cost_total)  # minimum cost of all trajectories
        cost_total_non_zero = _ensure_non_zero(costs=cost_total, beta=beta, factor=1/lambda_)

        eta = np.sum(cost_total_non_zero)

        omega = 1/eta * cost_total_non_zero

        U += [np.sum(omega * noise[:, t]) for t in range(TIMESTEPS)]

        env.env.state = state
        s, r, _, _ = env.step([U[0]])
        print("action taken: ", U[0], "cost received: ", -r)
        env.render()

        U = np.roll(U, -1)  # shift all elements to the left
        U[-1] = np.random.uniform(low=ACTION_LOW, high=ACTION_HIGH)
        state = env.env.state


if __name__ == "__main__":
    pendulum_MPPI()
