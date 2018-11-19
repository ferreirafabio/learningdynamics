import numpy as np
import tensorflow as tf
import time
from base.base_train import BaseTrain
from utils.utils import convert_dict_to_list_subdicts, get_all_images_from_gn_output, get_pos_ndarray_from_output
from models.singulation_graph import create_graphs_and_placeholders, create_feed_dict
from joblib import parallel_backend, Parallel, delayed



class SingulationTrainer(BaseTrain):
    def __init__(self, sess, model, train_data, valid_data, config, logger):
        super(SingulationTrainer, self).__init__(sess, model, train_data, valid_data, config, logger)

    def train_epoch(self):
        prefix = self.config.exp_name
        while True:
           try:
                _, _, cur_batch_it = self.train_batch(prefix)

                if cur_batch_it % self.config.model_save_step_interval == 1:
                    self.model.save(self.sess)
                if cur_batch_it % self.config.test_interval == 1:
                    print("Executing test batch")
                    self.test_batch(prefix)

           except tf.errors.OutOfRangeError:
               break

    def do_step(self, input_graph, target_graphs, feature, train=True):
        #exp_length = feature['experiment_length']

        #if exp_length != 2:
        #    return None, None

        feed_dict = create_feed_dict(self.model.input_ph, self.model.target_ph, input_graph, target_graphs)

        if train:
            data = self.sess.run({"step": self.model.step_op, "target": self.model.target_ph, "loss": self.model.loss_op_train,
                                  "outputs": self.model.output_ops_train, "pos_vel_loss": self.model.pos_vel_loss_ops_train
                                  }, feed_dict=feed_dict)

        else:
            data = self.sess.run({"step": self.model.step_op, "target": self.model.target_ph, "loss": self.model.loss_op_test,
                                  "outputs": self.model.output_ops_test, "pos_vel_loss": self.model.pos_vel_loss_ops_test
                                  }, feed_dict=feed_dict)

        return data['loss'], data['outputs'], data['pos_vel_loss']

    def train_batch(self, prefix):
        losses = []
        pos_vel_losses = []
        next_element = self.train_data.get_next_batch()
        features = self.sess.run(next_element)

        features = convert_dict_to_list_subdicts(features, self.config.train_batch_size)
        _, _, input_graphs_all_exp, target_graphs_all_exp = create_graphs_and_placeholders(config=self.config, batch_data=features,
                                                                                           batch_size=self.config.train_batch_size)

        start_time = time.time()
        last_log_time = start_time

        if self.config.parallel_batch_processing:
            with parallel_backend('threading', n_jobs=-3):
                losses, pos_vel_losses = Parallel()(delayed(self._do_step_parallel)(input_graphs_all_exp[i], target_graphs_all_exp[i],
                                                        features[i], losses) for i in range(self.config.train_batch_size))
        else:
            for i in range(self.config.train_batch_size):
                loss, _, pos_vel_loss = self.do_step(input_graphs_all_exp[i], target_graphs_all_exp[i], features[i])
                if loss is not None:
                    losses.append(loss)
                    pos_vel_losses.append(pos_vel_loss)

        the_time = time.time()
        elapsed_since_last_log = the_time - last_log_time

        self.sess.run(self.model.increment_cur_batch_tensor)
        cur_batch_it = self.model.cur_batch_tensor.eval(self.sess)

        if losses:
            batch_loss = np.mean(losses)
            pos_vel_batch_loss = np.mean(pos_vel_losses)
            print('batch: {:06d} loss: {:0.2f} pos_vel loss: {:0.2f} time (sec): {:0.2f}'.format(cur_batch_it, batch_loss, pos_vel_batch_loss, elapsed_since_last_log))
            summaries_dict = {prefix + '_loss': batch_loss, prefix + '_pos_vel_loss': pos_vel_batch_loss}

            self.logger.summarize(cur_batch_it, summaries_dict=summaries_dict, summarizer="train")
        else:
            batch_loss = 0
            pos_vel_batch_loss = 0

        return batch_loss, pos_vel_batch_loss, cur_batch_it

    def test_batch(self, prefix, log_position_displacements=False, log_vel_discplacements=False):

        losses = []
        pos_vel_losses = []
        output_for_summary = None
        next_element = self.test_data.get_next_batch()
        features = self.sess.run(next_element)

        features = convert_dict_to_list_subdicts(features, self.config.test_batch_size)
        _, _, input_graphs_all_exp, target_graphs_all_exp = create_graphs_and_placeholders(config=self.config, batch_data=features,
                                                                                         batch_size=self.config.test_batch_size)

        target_summaries_dict_rgb, target_summaries_dict_seg, target_summaries_dict_depth = {}, {}, {}
        predicted_summaries_dict_rgb, predicted_summaries_dict_seg, predicted_summaries_dict_depth = {}, {}, {}
        target_summaries_dict_global_img, target_summaries_dict_global_seg, target_summaries_dict_global_depth = {}, {}, {}

        start_time = time.time()
        last_log_time = start_time

        for i in range(self.config.test_batch_size):
            loss, outputs, pos_vel_loss = self.do_step(input_graphs_all_exp[i], target_graphs_all_exp[i], features[i], train=False)
            if loss is not None:
                losses.append(loss)
                pos_vel_losses.append(pos_vel_loss)
            ''' get the last not-None output '''
            if outputs is not None:
                output_for_summary = (outputs, i)

        the_time = time.time()
        elapsed_since_last_log = the_time - last_log_time

        self.sess.run(self.model.increment_cur_batch_tensor)
        cur_batch_it = self.model.cur_batch_tensor.eval(self.sess)

        if output_for_summary is not None:
            ''' returns n lists, each having an ndarray of shape (exp_length, w, h, c)  while n = number of objects '''

            images_rgb, images_seg, images_depth = get_all_images_from_gn_output(output_for_summary[0], self.config.depth_data_provided)
            features_index = output_for_summary[1]

            predicted_summaries_dict_seg = {prefix + '_predicted_seg_exp_id_{}_batch_{}_object_{}'.format(
                int(features[features_index]['experiment_id']), cur_batch_it, i): obj for i, obj in enumerate(images_seg)}

            predicted_summaries_dict_depth = {prefix + '_predicted_depth_exp_id_{}_batch_{}_object_{}'.format(
                int(features[features_index]['experiment_id']), cur_batch_it, i): obj for i, obj in enumerate(images_depth)}

            predicted_summaries_dict_rgb = {prefix + '_predicted_rgb_exp_id_{}_batch_{}_object_{}'.format(
                int(features[features_index]['experiment_id']), cur_batch_it, i): obj for i, obj in enumerate(images_rgb)}


            ''' get the ground truth images for comparison, [-3:] means 'get the last three manipulable objects '''
            n_manipulable_objects = features[features_index]['n_manipulable_objects']
            # shape [exp_length, n_objects, w, h, c] --> shape [n_objects, exp_length, w, h, c] --> split in n_objects lists -->
            # [n_split, n_objects, exp_length, ...]
            lists_obj_segs = np.split(np.swapaxes(features[features_index]['object_segments'], 0, 1)[-n_manipulable_objects:], n_manipulable_objects)

            target_summaries_dict_rgb = {prefix + '_target_rgb_exp_id_{}_batch_{}_object_{}'.format(
                features[features_index]['experiment_id'], cur_batch_it, i): np.squeeze(lst[..., :3], axis=0) for i, lst in enumerate(lists_obj_segs)}

            target_summaries_dict_seg = {prefix + '_target_seg_exp_id_{}_batch_{}_object_{}'.format(
                features[features_index]['experiment_id'], cur_batch_it, i): np.squeeze(np.expand_dims(lst[...,3], axis=4), axis=0) for i, lst in enumerate(lists_obj_segs)}

            target_summaries_dict_depth = {prefix + '_target_depth_exp_id_{}_batch_{}_object_{}'.format(
                features[features_index]['experiment_id'], cur_batch_it, i): np.squeeze(lst[..., -3:], axis=0) for i, lst in enumerate(lists_obj_segs)}

            target_summaries_dict_global_img = {prefix + '_target_global_img_exp_id_{}_batch_{}'.format(
                features[features_index]['experiment_id'], cur_batch_it): features[features_index]['img']}

            target_summaries_dict_global_seg = {prefix + '_target_global_seg_exp_id_{}_batch_{}'.format(
                features[features_index]['experiment_id'], cur_batch_it): np.expand_dims(features[features_index]['seg'], axis=4)}

            target_summaries_dict_global_depth = {prefix + '_target_global_depth_exp_id_{}_batch{}'.format(cur_batch_it,
                features[features_index]['experiment_id'], cur_batch_it): features[features_index]['depth']}

            if log_position_displacements:
                pos_array_predicted = get_pos_ndarray_from_output(output_for_summary)
                pos_array_target = features[features_index]['objpos']
                # todo: check whether both arrays come from the same input-target-pair
                raise NotImplementedError


        if losses:
            batch_loss = np.mean(losses)
            pos_vel_batch_loss = np.mean(pos_vel_losses)
            print('current test loss on batch: {:0.2f} pos_vel loss: {:0.2f} time (sec): {:0.2f}'.format(batch_loss, pos_vel_batch_loss, elapsed_since_last_log))

            summaries_dict = {prefix + '_loss': batch_loss, prefix + '_pos_vel_loss': pos_vel_batch_loss}
            summaries_dict = {
                **predicted_summaries_dict_rgb, **predicted_summaries_dict_seg, **predicted_summaries_dict_depth,
                **summaries_dict,
                **target_summaries_dict_rgb, **target_summaries_dict_seg, **target_summaries_dict_depth,
                **target_summaries_dict_global_img, **target_summaries_dict_global_seg, **target_summaries_dict_global_depth
            }

            cur_batch_it = self.model.cur_batch_tensor.eval(self.sess)
            self.logger.summarize(cur_batch_it, summaries_dict=summaries_dict, summarizer="test")

        else:
            batch_loss = 0
            pos_vel_batch_loss = 0

        return batch_loss, pos_vel_batch_loss


    def _do_step_parallel(self, input_graphs_all_exp, target_graphs_all_exp, features, losses, pos_vel_losses):
        loss, _, pos_vel_loss = self.do_step(input_graphs_all_exp, target_graphs_all_exp, features, train=True)
        if loss is not None:
            losses.append(loss)
            pos_vel_losses.append(pos_vel_loss)
        return losses, pos_vel_loss
