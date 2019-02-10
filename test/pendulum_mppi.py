import gym
import numpy as np

from base.base_mpc import MPPI_gym

ENV_NAME = "Pendulum-v0"

TIMESTEPS = 20  # T
N_SAMPLES = 1000  # K
ACTION_LOW = -2.0
ACTION_HIGH = 2.0

noise_mu = 0
noise_sigma = 10
lambda_ = 1


if __name__ == "__main__":
    U = np.random.uniform(low=ACTION_LOW, high=ACTION_HIGH, size=TIMESTEPS)  # pendulum joint effort in (-2, +2)

    env = gym.make(ENV_NAME)
    mppi_gym = MPPI_gym(env=env, K=N_SAMPLES, T=TIMESTEPS, U=U, lambda_=lambda_, noise_mu=noise_mu, noise_sigma=noise_sigma, u_init=0, noise_gaussian=True)
    mppi_gym.control(1000, parallel=True)
