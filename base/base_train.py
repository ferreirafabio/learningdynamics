import tensorflow as tf
from models.singulation_graph import create_placeholders
from utils.conversions import convert_dict_to_list_subdicts
from utils.utils import make_all_runnable_in_session
from models.loss_functions import create_loss_ops


class BaseTrain:
    def __init__(self, sess, model, train_data, test_data, config, logger):
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        self.train_data = train_data
        self.test_data = test_data

        self.initialize_train_model()
        self.initialize_test_model()

        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

        self.model.init_saver()

    def train(self):
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.n_epochs + 1, 1):
            self.train_epoch()
            self.sess.run(self.model.increment_cur_epoch_tensor)

    def test(self):
        self.test_epoch()

    def test_rollouts(self):
        self.test_rollouts()

    def test_epoch(self):
        raise NotImplementedError

    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def do_step(self, input_graphs_all_exp, target_graphs_all_exp, features, train_flag):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError

    def initialize_train_model(self):
        next_element = self.train_data.get_next_batch()
        features = self.sess.run(next_element)
        features = convert_dict_to_list_subdicts(features, self.config.train_batch_size)

        input_ph, target_ph, input_ctrl_ph = create_placeholders(self.config, features)
        input_ph, target_ph = make_all_runnable_in_session(input_ph, target_ph)

        self.model.input_ph = input_ph
        self.model.target_ph = target_ph
        self.model.input_ctrl_ph = input_ctrl_ph

        self.model.is_training = True
        self.model.output_ops_train = self.model(self.model.input_ph, self.model.input_ctrl_ph, self.config.n_rollouts, self.model.is_training)
        total_loss_ops, loss_ops_img, loss_ops_iou, loss_ops_velocity, loss_ops_position, loss_ops_distance = create_loss_ops(self.config, self.model.target_ph, self.model.output_ops_train)
        ''' remove all inf values --> correspond to padded entries '''
        self.model.loss_op_train_total = tf.reduce_mean(tf.boolean_mask(total_loss_ops, tf.logical_not(tf.is_inf(total_loss_ops))))
        self.model.loss_ops_train_img = tf.reduce_mean(tf.boolean_mask(loss_ops_img, tf.logical_not(tf.is_inf(loss_ops_img))))  # just for summary, is already included in loss_op_train
        self.model.loss_ops_train_iou = tf.reduce_mean(tf.boolean_mask(loss_ops_iou, tf.logical_not(tf.is_inf(loss_ops_iou))))
        self.model.loss_ops_train_velocity = tf.reduce_mean(tf.boolean_mask(loss_ops_velocity, tf.logical_not(tf.is_inf(loss_ops_velocity))))
        self.model.loss_ops_train_position = tf.reduce_mean(tf.boolean_mask(loss_ops_position, tf.logical_not(tf.is_inf(loss_ops_position))))
        self.model.loss_ops_train_distance = tf.reduce_mean(tf.boolean_mask(loss_ops_distance, tf.logical_not(tf.is_inf(loss_ops_distance))))

        self.model.step_op = self.model.optimizer.minimize(self.model.loss_op_train_total, global_step=self.model.global_step_tensor)

    def initialize_test_model(self):
        next_element = self.test_data.get_next_batch()
        features = self.sess.run(next_element)
        features = convert_dict_to_list_subdicts(features, self.config.test_batch_size)

        input_ph, target_ph, input_ctrl_ph = create_placeholders(self.config, features)
        input_ph, target_ph = make_all_runnable_in_session(input_ph, target_ph)

        self.model.input_ph_test = input_ph
        self.model.target_ph_test = target_ph
        self.model.input_ctrl_ph_test = input_ctrl_ph

        self.model.is_training = False
        self.model.output_ops_test = self.model(self.model.input_ph_test, self.model.input_ctrl_ph_test, self.config.n_rollouts, self.model.is_training)
        total_loss_ops_test, loss_ops_test_img, loss_ops_test_iou, loss_ops_test_velocity, loss_ops_test_position, loss_ops_test_distance = create_loss_ops(self.config, self.model.target_ph_test, self.model.output_ops_test)

        ''' remove all inf values --> correspond to padded entries '''
        self.model.loss_op_test_total = tf.reduce_mean(tf.boolean_mask(total_loss_ops_test, tf.logical_not(tf.is_inf(total_loss_ops_test)))) # just for summary, is already included in loss_op_train
        self.model.loss_ops_test_img = tf.reduce_mean(tf.boolean_mask(loss_ops_test_img, tf.logical_not(tf.is_inf(loss_ops_test_img))))
        self.model.loss_ops_test_iou = tf.reduce_mean(tf.boolean_mask(loss_ops_test_iou, tf.logical_not(tf.is_inf(loss_ops_test_iou))))
        self.model.loss_ops_test_velocity = tf.reduce_mean(tf.boolean_mask(loss_ops_test_velocity, tf.logical_not(tf.is_inf(loss_ops_test_velocity))))
        self.model.loss_ops_test_position = tf.reduce_mean(tf.boolean_mask(loss_ops_test_position, tf.logical_not(tf.is_inf(loss_ops_test_position))))
        self.model.loss_ops_test_distance = tf.reduce_mean(tf.boolean_mask(loss_ops_test_distance, tf.logical_not(tf.is_inf(loss_ops_test_distance))))

