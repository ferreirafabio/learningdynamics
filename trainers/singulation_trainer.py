import numpy as np
import tensorflow as tf
from base.base_train import BaseTrain
from utils.utils import convert_dict_to_list_subdicts, get_all_images_from_gn_output
from models.singulation_graph import create_graphs_and_placeholders, create_feed_dict


class SingulationTrainer(BaseTrain):
    def __init__(self, sess, model, train_data, valid_data, config, logger):
        super(SingulationTrainer, self).__init__(sess, model, train_data, valid_data, config, logger)

    def train_epoch(self):
        prefix = self.config.exp_name
        while True:
           try:
                _, cur_it = self.train_batch(prefix)
                if cur_it % self.config.model_save_step_interval == 1:
                    self.model.save(self.sess)
                if cur_it % self.config.test_interval == 1:
                    print("Executing test batch")
                    self.test_batch(prefix)

           except tf.errors.OutOfRangeError:
               break

    def do_step(self, input_graph, target_graphs, feature, train=True):
        exp_length = feature['experiment_length']

        #if exp_length != 2:
        #    return None, None

        feed_dict = create_feed_dict(self.model.input_ph, self.model.target_ph, input_graph, target_graphs)

        if train:
            data = self.sess.run({"step": self.model.step_op, "target": self.model.target_ph, "loss": self.model.loss_op_train,
                "outputs": self.model.output_ops_train}, feed_dict=feed_dict)

        else:
            data = self.sess.run({"step": self.model.step_op, "target": self.model.target_ph, "loss": self.model.loss_op_test,
                                  "outputs": self.model.output_ops_test}, feed_dict=feed_dict)

        return data['loss'], data['outputs']

    def train_batch(self, prefix):
        losses = []
        next_element = self.train_data.get_next_batch()
        features = self.sess.run(next_element)

        features = convert_dict_to_list_subdicts(features, self.config.train_batch_size)
        _, _, input_graphs_all_exp, target_graphs_all_exp = create_graphs_and_placeholders(config=self.config, batch_data=features,
                                                                                         batch_size=self.config.train_batch_size)
        for i in range(self.config.train_batch_size):
            loss, _ = self.do_step(input_graphs_all_exp[i], target_graphs_all_exp[i], features[i])
            if loss is not None:
                losses.append(loss)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        if losses:
            batch_loss = np.mean(losses)
            print('step {:06d} loss (batch) {:0.2f}'.format(cur_it, batch_loss))
            summaries_dict = {prefix + '_loss': batch_loss}
            self.logger.summarize(cur_it, summaries_dict=summaries_dict, summarizer="train")
        else:
            batch_loss = 0

        return batch_loss, cur_it

    def test_batch(self, prefix):
        losses = []
        output_for_summary = None
        next_element = self.test_data.get_next_batch()
        features = self.sess.run(next_element)

        features = convert_dict_to_list_subdicts(features, self.config.test_batch_size)
        _, _, input_graphs_all_exp, target_graphs_all_exp = create_graphs_and_placeholders(config=self.config, batch_data=features,
                                                                                         batch_size=self.config.test_batch_size)

        target_summaries_dict_rgb, target_summaries_dict_seg, target_summaries_dict_depth = {}, {}, {}
        predicted_summaries_dict_rgb, predicted_summaries_dict_seg, predicted_summaries_dict_depth = {}, {}, {}
        target_summaries_dict_global_img, target_summaries_dict_global_seg, target_summaries_dict_global_depth = {}, {}, {}

        cur_it = self.model.global_step_tensor.eval(self.sess)
        for i in range(self.config.test_batch_size):
            loss, outputs = self.do_step(input_graphs_all_exp[i], target_graphs_all_exp[i], features[i], train=False)

            if loss is not None:
                losses.append(loss)
            ''' get the last not None output '''
            if outputs is not None:
                output_for_summary = (outputs, i)

        if output_for_summary is not None:
            ''' returns n lists, each having an ndarray of shape (exp_length, w, h, c)  while n = number of objects '''

            images_rgb, images_seg, images_depth = get_all_images_from_gn_output(output_for_summary[0], self.config.depth_data_provided)
            features_index = output_for_summary[1]

            predicted_summaries_dict_seg = {prefix + '_predicted_img_seg_exp_id_{}_object_{}'.format(
                int(features[features_index]['experiment_id']), i): obj for i, obj in enumerate(images_seg)}

            predicted_summaries_dict_depth = {prefix + '_predicted_img_depth_exp_id_{}_object_{}'.format(
                int(features[features_index]['experiment_id']), i): obj for i, obj in enumerate(images_depth)}

            predicted_summaries_dict_rgb = {prefix + '_predicted_img_rgb_exp_id_{}_object_{}'.format(
                int(features[features_index]['experiment_id']), i): obj for i, obj in enumerate(images_rgb)}


            ''' get the ground truth images for comparison, [-3:] means 'get the last three manipulable objects '''
            n_manipulable_objects = features[features_index]['n_manipulable_objects']
            # shape [exp_length, n_objects, w, h, c] --> shape [n_objects, exp_length, w, h, c] --> split in n_objects lists -->
            # [n_split, n_objects, exp_length, ...]
            lists_obj_segs = np.split(np.swapaxes(features[features_index]['object_segments'], 0, 1)[-n_manipulable_objects:], n_manipulable_objects)

            target_summaries_dict_rgb = {prefix + '_target_img_rgb_exp_id_{}_object_{}'.format(
                features[features_index]['experiment_id'], i): np.squeeze(lst[..., :3], axis=0) for i, lst in enumerate(lists_obj_segs)}

            target_summaries_dict_seg = {prefix + '_target_img_seg_exp_id_{}_object_{}'.format(
                features[features_index]['experiment_id'], i): np.squeeze(np.expand_dims(lst[...,3], axis=4), axis=0) for i, lst in enumerate(lists_obj_segs)}

            target_summaries_dict_depth = {prefix + '_target_img_depth_exp_id_{}_object_{}'.format(
                features[features_index]['experiment_id'], i): np.squeeze(lst[..., -3:], axis=0) for i, lst in enumerate(lists_obj_segs)}

            target_summaries_dict_global_img = {prefix + '_target_global_img_exp_id_{}'.format(
                features[features_index]['experiment_id']): features[features_index]['img']}

            target_summaries_dict_global_seg = {prefix + '_target_global_seg_exp_id_{}'.format(
                features[features_index]['experiment_id']): np.expand_dims(features[features_index]['seg'], axis=4)}

            target_summaries_dict_global_depth = {prefix + '_target_global_depth_exp_id_{}'.format(
                features[features_index]['experiment_id']): features[features_index]['depth']}

            # todo: show global image

        if losses:
            batch_loss = np.mean(losses)
            print('step {:06d} loss (test batch) {:0.2f}'.format(cur_it, batch_loss))

            summaries_dict = {prefix + '_loss': batch_loss}
            summaries_dict = {
                **predicted_summaries_dict_rgb, **predicted_summaries_dict_seg, **predicted_summaries_dict_depth,
                **summaries_dict,
                **target_summaries_dict_rgb, **target_summaries_dict_seg, **target_summaries_dict_depth,
                **target_summaries_dict_global_img, **target_summaries_dict_global_seg, **target_summaries_dict_global_depth
            }

            self.logger.summarize(cur_it, summaries_dict=summaries_dict, summarizer="test")

        else:
            batch_loss = 0

        return batch_loss, cur_it


