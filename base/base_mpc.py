import tensorflow as tf
import numpy as np
import time
import multiprocessing
from joblib import parallel_backend, Parallel, delayed
from multiprocessing import Pool


class MPC:
    def __init__(self, sess, forward_model, train_data, test_data, config, logger, N=10):
        self.forward_model = forward_model
        self.logger = logger
        self.config = config
        self.sess = sess
        self.train_data = train_data
        self.test_data = test_data
        self.is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
        self.N = N

    def tf_eval(self):
        raise NotImplementedError

    def solve(self, state):
        raise NotImplementedError


class MPPI:
    def __init__(self, K, T, U, lambda_=1.0, noise_mu=0, noise_sigma=1, u_init=1, noise_gaussian=True):
        self.K = K  # N_SAMPLES
        self.T = T  # TIMESTEPS
        self.lambda_ = lambda_
        self.noise_mu = noise_mu
        self.noise_sigma = noise_sigma
        self.U = U
        self.u_init = u_init
        self.cost_total = np.zeros(shape=(self.K))

        if noise_gaussian:
            self.noise = np.random.normal(loc=self.noise_mu, scale=self.noise_sigma, size=(self.K, self.T))
        else:
            self.noise = np.full(shape=(self.K, self.T), fill_value=0.9)

    def _ensure_non_zero(self, cost, beta, factor):
        return np.exp(-factor * (cost - beta))


class MPPI_gym_parallel(MPPI):
    """ MMPI according to algorithm 2 in Williams et al., 2017
        'Information Theoretic MPC for Model-Based Reinforcement Learning' """

    def __init__(self, env, K, T, U, lambda_=1.0, noise_mu=1, noise_sigma=100, u_init=1, x_init=None, noise_gaussian=True, downward_start=True):
        super(MPPI_gym_parallel, self).__init__(K=K, T=T, U=U, lambda_=lambda_, noise_mu=noise_mu, noise_sigma=noise_sigma,
                                                u_init=u_init, x_init=x_init, noise_gaussian=noise_gaussian)
        self.env = env
        self.env.reset()
        if downward_start:
            self.env.env.state = [np.pi, 1]
        self.x_init = env.env.state


    def _control(self, _):
        self._compute_total_cost()
        beta = np.min(self.cost_total)  # minimum cost of all trajectories
        cost_total_non_zero = self._ensure_non_zero(cost=self.cost_total, beta=beta, factor=1 / self.lambda_)

        eta = np.sum(cost_total_non_zero)
        omega = 1 / eta * cost_total_non_zero

        self.U += [np.sum(omega * self.noise[:, t]) for t in range(self.T)]

        self.env.env.state = self.x_init
        s, r, _, _ = self.env.step([self.U[0]])
        print("action taken: ", self.U[0], "cost received: ", -r)
        self.env.render()

        self.U = np.roll(self.U, -1)  # shift all elements to the left
        self.U[-1] = self.u_init
        self.x_init = self.env.env.state


    def _compute_total_cost(self):
        for k in range(self.K):
            self.env.env.state = self.x_init
            for t in range(self.T):
                perturbed_action_t = self.U[t] + self.noise[k, t]
                _, reward, _, _ = self.env.step([perturbed_action_t])
                self.cost_total[k] += -reward


    def control(self, iter=1000, parallel=False):
        if parallel:
            with Pool(5) as p:
                print(p.map(self._control, range(iter)))
        else:
            for _ in range(iter):
                self._control()




class MPPI_gym(MPPI):
    """ MMPI according to algorithm 2 in Williams et al., 2017
        'Information Theoretic MPC for Model-Based Reinforcement Learning' """

    def __init__(self, env, K, T, U, lambda_=1.0, noise_mu=1, noise_sigma=100, u_init=1, x_init=None, noise_gaussian=True, downward_start=True):
        super(MPPI_gym, self).__init__(K=K, T=T, U=U, lambda_=lambda_, noise_mu=noise_mu, noise_sigma=noise_sigma,
                                       u_init=u_init, noise_gaussian=noise_gaussian)
        self.env = env
        self.env.reset()
        if downward_start:
            self.env.env.state = [np.pi, 1]
        self.x_init = self.env.env.state

    def _compute_total_cost(self, k):
        self.env.env.state = self.x_init
        for t in range(self.T):
            perturbed_action_t = self.U[t] + self.noise[k, t]
            _, reward, _, _ = self.env.step([perturbed_action_t])
            self.cost_total[k] += -reward

    def control(self, iter=1000, parallel=False):
        for _ in range(iter):
            start_time = time.time()
            last_log_time = start_time

            if parallel:
                with parallel_backend('threading', n_jobs=multiprocessing.cpu_count()):
                    Parallel()(delayed(self._compute_total_cost)(k) for k in range(self.K))
            else:
                for k in range(self.K):
                    self._compute_total_cost(k)

            the_time = time.time()
            elapsed_since_last_log = the_time - last_log_time

            beta = np.min(self.cost_total)  # minimum cost of all trajectories
            cost_total_non_zero = self._ensure_non_zero(cost=self.cost_total, beta=beta, factor=1/self.lambda_)

            eta = np.sum(cost_total_non_zero)
            omega = 1/eta * cost_total_non_zero

            self.U += [np.sum(omega * self.noise[:, t]) for t in range(self.T)]

            self.env.env.state = self.x_init
            s, r, _, _ = self.env.step([self.U[0]])
            print("action taken: {:.2f} cost received: {:.2f} duration: {:.2f}".format(self.U[0], -r, elapsed_since_last_log))
            self.env.render()

            self.U = np.roll(self.U, -1)  # shift all elements to the left
            self.U[-1] = self.u_init
            self.cost_total[:] = 0
            self.x_init = self.env.env.state



