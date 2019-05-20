import tensorflow as tf
from models.singulation_graph import create_placeholders
from utils.conversions import convert_dict_to_list_subdicts
from utils.utils import make_all_runnable_in_session
from models.loss_functions import create_loss_ops, create_baseline_loss_ops


class BaseTrain:
    def __init__(self, sess, model, train_data, test_data, config, logger, only_test=False):
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        self.train_data = train_data
        self.test_data = test_data

        if not self.config.use_baseline_auto_predictor:
            if not only_test:
                self.initialize_train_model()
            self.initialize_test_model()
        else:
            self.initialize_model()

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

    def initialize_model(self):
        self.in_image_tf = tf.placeholder(tf.float32, [None, 120, 160, 3], 'in_image')
        self.in_segxyz_tf = tf.placeholder(tf.float32, [None, 120, 160, 4], 'in_xyzseg')
        self.in_control_tf = tf.placeholder(tf.float32, [None, 6], 'in_control')

        self.gt_predictions = tf.placeholder(tf.float32, [None, 120, 160], 'gt_predictions')

        self.is_training = tf.placeholder(tf.bool, shape=(), name="is_training")

        #self.batch_size = tf.placeholder(tf.int32, [self.config.train_batch_size])

        if "predictor_extended" in self.config.model_zoo_file:
            self.gt_latent_vectors = tf.placeholder(tf.float32, [None, 256], 'gt_encoding_vectors')

            self.out_predictions, self.in_rgb_seg_xyz, self.out_latent_vectors, self.debug_in_control = \
                self.model.cnnmodel(in_rgb=self.in_image_tf,
                                    in_segxyz=self.in_segxyz_tf,
                                    in_control=self.in_control_tf,
                                    is_training=self.is_training,
                                    n_predictions=self.config.n_predictions)

        elif "baseline_auto_predictor_multistep" in self.config.model_zoo_file:
            self.out_predictions, self.in_rgb_seg_xyz, self.out_latent_vectors, self.debug_in_control = \
                self.model.cnnmodel(in_rgb=self.in_image_tf,
                                    in_segxyz=self.in_segxyz_tf,
                                    in_control=self.in_control_tf,
                                    is_training=self.is_training,
                                    n_predictions=self.config.n_predictions)

        elif "baseline_auto_encoder" in self.config.model_zoo_file:
            self.out_predictions, self.in_rgb_seg_xyz, self.encoder_outputs = self.model.cnnmodel(in_rgb=self.in_image_tf,
                                                                                                      in_segxyz=self.in_segxyz_tf,
                                                                                                      in_control=self.in_control_tf,
                                                                                                      is_training=self.is_training)
        else:
            self.out_predictions, self.in_rgb_seg_xyz = self.model.cnnmodel(in_rgb=self.in_image_tf,
                                                                            in_segxyz=self.in_segxyz_tf,
                                                                            in_control=self.in_control_tf,
                                                                            is_training=self.is_training)

        self.out_prediction_softmax = tf.nn.softmax(self.out_predictions)[:, :, :, 1]

        if "auto_encoding" in self.config.loss_type:
            self.model.loss_op = create_baseline_loss_ops(config=self.config,
                                                          gt_predictions=None,
                                                          out_predictions=None,
                                                          gt_reconstructions=self.gt_predictions,
                                                          out_reconstructions=self.out_predictions,
                                                          gt_latent_vectors=None,
                                                          out_latent_vectors=None,
                                                          loss_type=self.config.loss_type)

        elif "prediction_perception_loss" in self.config.loss_type:
            self.model.loss_op, self.model.img_loss, self.model.perception_loss = create_baseline_loss_ops(config=self.config,
                                                          gt_predictions=self.gt_predictions,
                                                          out_predictions=self.out_predictions,
                                                          gt_reconstructions=None,
                                                          out_reconstructions=None,
                                                          gt_latent_vectors=self.gt_latent_vectors,
                                                          out_latent_vectors=self.out_latent_vectors,
                                                          loss_type=self.config.loss_type)
        else:
            self.model.loss_op = create_baseline_loss_ops(config=self.config,
                                                          gt_predictions=self.gt_predictions,
                                                          out_predictions=self.out_predictions,
                                                          gt_reconstructions=None,
                                                          out_reconstructions=None,
                                                          gt_latent_vectors=None,
                                                          out_latent_vectors=None,
                                                          loss_type=self.config.loss_type)

        self.model.train_op = self.model.optimizer.minimize(self.model.loss_op, global_step=self.model.global_step_tensor)

    def initialize_train_model(self):
        next_element = self.train_data.get_next_batch()
        features = self.sess.run(next_element)
        features = convert_dict_to_list_subdicts(features, self.config.train_batch_size)

        if self.config.batch_processing:
            input_ph, target_ph = create_placeholders(self.config, features, batch_processing=True)
        else:
            input_ph, target_ph = create_placeholders(self.config, features[0], batch_processing=False)

        self.model.input_ph = input_ph
        self.model.target_ph = target_ph
        self.model.is_training = True
        self.model.output_ops_train, self.model.latent_core_output_init_img_train, self.model.latent_encoder_output_init_img_train = self.model(self.model.input_ph, self.model.target_ph,
                                                 1, self.model.is_training)

        total_loss_ops, loss_ops_img, loss_ops_iou, loss_ops_velocity, loss_ops_position, loss_ops_distance, loss_ops_global = create_loss_ops(self.config, self.model.target_ph, self.model.output_ops_train)
        ''' remove all inf values --> correspond to padded entries '''
        self.model.loss_op_train_total = total_loss_ops
        self.model.loss_ops_train_img = loss_ops_img  # just for summary, is already included in loss_op_train
        self.model.loss_ops_train_iou = loss_ops_iou
        self.model.loss_ops_train_velocity = loss_ops_velocity
        self.model.loss_ops_train_position = loss_ops_position
        self.model.loss_ops_train_distance = loss_ops_distance
        self.model.loss_ops_train_global = loss_ops_global
        #self.model.train_logits = logits

        self.model.step_op = self.model.optimizer.minimize(self.model.loss_op_train_total, global_step=self.model.global_step_tensor)

    def initialize_test_model(self):
        next_element = self.test_data.get_next_batch()
        features = self.sess.run(next_element)
        features = convert_dict_to_list_subdicts(features, self.config.test_batch_size)

        input_ph, target_ph = create_placeholders(self.config, features[0], batch_processing=False)

        self.model.input_ph_test = input_ph
        self.model.target_ph_test = target_ph

        self.model.is_training = False
        self.model.output_ops_test, self.model.latent_core_output_init_img_test, self.model.latent_encoder_output_init_img_test = self.model(self.model.input_ph_test,
                                                self.model.target_ph_test, 1,
                                                self.model.is_training)

        total_loss_ops_test, loss_ops_test_img, loss_ops_test_iou, loss_ops_test_velocity, loss_ops_test_position, loss_ops_test_distance, loss_ops_test_global = create_loss_ops(self.config, self.model.target_ph_test, self.model.output_ops_test)

        ''' remove all inf values --> correspond to padded entries '''
        self.model.loss_op_test_total = total_loss_ops_test
        self.model.loss_ops_test_img = loss_ops_test_img
        self.model.loss_ops_test_iou = loss_ops_test_iou
        self.model.loss_ops_test_velocity = loss_ops_test_velocity
        self.model.loss_ops_test_position = loss_ops_test_position
        self.model.loss_ops_test_distance = loss_ops_test_distance
        self.model.loss_ops_test_global = loss_ops_test_global
        #self.model.test_logits = logits

