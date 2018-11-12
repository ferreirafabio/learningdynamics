import tensorflow as tf
from models.singulation_graph import create_placeholders
from utils.utils import convert_dict_to_list_subdicts
from utils.utils import make_all_runnable_in_session

class BaseTrain:
    def __init__(self, sess, model, data, config, logger):
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        self.data = data

        self.initialize_train_model()

        # todo test model
        self.initialize_train_model()

        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)


    def train(self):
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.n_epochs + 1, 1):
            self.train_epoch()
            self.sess.run(self.model.increment_cur_epoch_tensor)

    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_step(self):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError


    def initialize_train_model(self):
        next_element = self.data.get_next_batch()
        features = self.sess.run(next_element)
        features = convert_dict_to_list_subdicts(features, self.config.train_batch_size)

        input_ph, target_ph = create_placeholders(self.config, features)
        input_ph, target_ph = make_all_runnable_in_session(input_ph, target_ph)

        self.model.input_ph = input_ph
        self.model.target_ph = target_ph

        self.model.output_ops_train = self.model(self.model.input_ph, 10) # todo
        loss_ops_train = self.model.create_loss_ops(self.model.target_ph, self.model.output_ops_train)
        self.model.loss_op_train = tf.reduce_mean(loss_ops_train)
        self.model.step_op = self.model.optimizer.minimize(self.model.loss_op_train, global_step=self.model.global_step_tensor)


    def initialize_test_model(self):
        assert self.model.input_ph is not None
        assert self.model.target_ph is not None

        self.model.output_ops_test = self.model(self.model.input_ph, 10) # todo
        loss_ops_test = self.model.create_loss_ops(self.model.target_ph, self.model.output_ops_test)
        self.model.loss_op_test = tf.reduce_mean(loss_ops_test)
        self.model.step_op = self.model.optimizer.minimize(self.model.loss_op_test, global_step=self.model.global_step_tensor)
