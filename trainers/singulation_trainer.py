import numpy as np
import tensorflow as tf
import time
import os
from base.base_train import BaseTrain
from utils.utils import convert_dict_to_list_subdicts, get_all_images_from_gn_output, get_pos_ndarray_from_output, create_dir,\
                        save_to_gif_from_dict
from utils.tensorflow import create_predicted_summary_dicts, create_target_summary_dicts
from models.singulation_graph import create_graphs, create_feed_dict
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
                    self.test_batch(prefix, export_images=self.config.export_test_images)

           except tf.errors.OutOfRangeError:
               break

    def do_step(self, input_graph, target_graphs, feature, train=True):
        feed_dict = create_feed_dict(self.model.input_ph, self.model.target_ph, input_graph, target_graphs)


        if train:
            #feed_dict['is_training'] = True
            run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
            data = self.sess.run({"step": self.model.step_op, "target": self.model.target_ph, "loss": self.model.loss_op_train,
                                  "outputs": self.model.output_ops_train, "pos_vel_loss": self.model.pos_vel_loss_ops_train
                                  }, options=run_options, feed_dict=feed_dict)

        else:
            #feed_dict['is_training'] = False
            data = self.sess.run({"target": self.model.target_ph, "loss": self.model.loss_op_test,
                                  "outputs": self.model.output_ops_test, "pos_vel_loss": self.model.pos_vel_loss_ops_test
                                  }, feed_dict=feed_dict)

        del feed_dict

        return data['loss'], data['outputs'], data['pos_vel_loss']

    def train_batch(self, prefix):
        losses = []
        pos_vel_losses = []

        next_element = self.train_data.get_next_batch()
        features = self.sess.run(next_element)

        features = convert_dict_to_list_subdicts(features, self.config.train_batch_size)
        input_graphs_all_exp, target_graphs_all_exp = create_graphs(config=self.config, batch_data=features,
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
            print('batch: {:<10} loss: {:<12.2f} pos_vel loss: {:<10.2f} batch processing time (sec): {:<10.2f} '
                .format(
                cur_batch_it, batch_loss, pos_vel_batch_loss, elapsed_since_last_log)
            )
            summaries_dict = {prefix + '_loss': batch_loss, prefix + '_pos_vel_loss': pos_vel_batch_loss}
            self.logger.summarize(cur_batch_it, summaries_dict=summaries_dict, summarizer="train")
        else:
            batch_loss = 0
            pos_vel_batch_loss = 0

        return batch_loss, pos_vel_batch_loss, cur_batch_it

    def test_batch(self, prefix, log_position_displacements=False, log_vel_discplacements=False, export_images=False):
        losses = []
        pos_vel_losses = []
        output_for_summary = None
        next_element = self.test_data.get_next_batch()
        features = self.sess.run(next_element)

        features = convert_dict_to_list_subdicts(features, self.config.test_batch_size)
        input_graphs_all_exp, target_graphs_all_exp = create_graphs(config=self.config, batch_data=features,
                                                                                         batch_size=self.config.test_batch_size)

        summaries_dict_images = {}
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

        cur_batch_it = self.model.cur_batch_tensor.eval(self.sess)

        if output_for_summary is not None:
            ''' returns n lists, each having an ndarray of shape (exp_length, w, h, c)  while n = number of objects '''
            images_rgb, images_seg, images_depth = get_all_images_from_gn_output(output_for_summary[0], self.config.depth_data_provided)
            features_index = output_for_summary[1]

            predicted_summaries_dict_seg, predicted_summaries_dict_depth, predicted_summaries_dict_rgb = create_predicted_summary_dicts(
                images_seg,
                images_depth,
                images_rgb,
                prefix=prefix,
                features=features,
                features_index=features_index,
                cur_batch_it=cur_batch_it
            )

            target_summaries_dict_rgb, target_summaries_dict_seg, target_summaries_dict_depth, target_summaries_dict_global_img, \
            target_summaries_dict_global_seg, target_summaries_dict_global_depth = create_target_summary_dicts(
                prefix=prefix,
                features=features,
                features_index=features_index,
                cur_batch_it=cur_batch_it
            )



            if log_position_displacements:
                pos_array_predicted = get_pos_ndarray_from_output(output_for_summary)
                pos_array_target = features[features_index]['objpos']
                # todo: check whether both arrays come from the same input-target-pair
                raise NotImplementedError



        if losses:
            batch_loss = np.mean(losses)
            pos_vel_batch_loss = np.mean(pos_vel_losses)
            print('test batch loss: {:<12.2f} pos_vel loss: {:<12.2f} time (sec): {:<12.2f}'.format(batch_loss, pos_vel_batch_loss, elapsed_since_last_log))

            summaries_dict = {prefix + '_loss': batch_loss, prefix + '_pos_vel_loss': pos_vel_batch_loss}
            summaries_dict_images = {
                **predicted_summaries_dict_rgb, **predicted_summaries_dict_seg, **predicted_summaries_dict_depth,
                **target_summaries_dict_rgb, **target_summaries_dict_seg, **target_summaries_dict_depth,
                **target_summaries_dict_global_img, **target_summaries_dict_global_seg, **target_summaries_dict_global_depth
            }

            summaries_dict = {**summaries_dict, **summaries_dict_images}

            cur_batch_it = self.model.cur_batch_tensor.eval(self.sess)
            self.logger.summarize(cur_batch_it, summaries_dict=summaries_dict, summarizer="test")

        else:
            batch_loss = 0
            pos_vel_batch_loss = 0


        if export_images:
            dir_path = create_dir(os.path.join("../experiments", prefix), "summary_images_batch_{}".format(cur_batch_it))
            save_to_gif_from_dict(image_dicts=summaries_dict_images, destination_path=dir_path, fps=10)

        return batch_loss, pos_vel_batch_loss


    def _do_step_parallel(self, input_graphs_all_exp, target_graphs_all_exp, features, losses, pos_vel_losses):
        loss, _, pos_vel_loss = self.do_step(input_graphs_all_exp, target_graphs_all_exp, features, train=True)
        if loss is not None:
            losses.append(loss)
            pos_vel_losses.append(pos_vel_loss)
        return losses, pos_vel_loss
