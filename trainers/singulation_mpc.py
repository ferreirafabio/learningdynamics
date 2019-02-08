from base.base_mpc import MPC


class MPCTrainer(MPC):
    def __init__(self, sess, model, train_data, valid_data, config, logger, N=10):
        super(MPCTrainer, self).__init__(sess, model, train_data, valid_data, config, logger, N=N)


    def tf_eval(self):
        raise NotImplementedError

    def solve(self, state):
        raise NotImplementedError