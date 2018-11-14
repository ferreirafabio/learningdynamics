import numpy as np
import tensorflow as tf
from base.base_train import BaseTrain
from utils.utils import convert_dict_to_list_subdicts, get_all_images_from_gn_output
from models.singulation_graph import create_graphs_and_placeholders, create_feed_dict


class SingulationTrainer(BaseTrain):
    def __init__(self, sess, model, train_data, valid_data, config, logger):
        super(SingulationTrainer, self).__init__(sess, model, train_data, valid_data, config, logger)

    def train_epoch(self):
        while True:
           try:
                _, cur_it = self.train_batch()

                if cur_it % self.config.model_save_interval == 0:
                    self.model.save(self.sess)
                if cur_it % self.config.test_interval == 0:
                    print("Executing test batch")
                    self.test_batch()

           except tf.errors.OutOfRangeError:
               break

    def train_batch(self):
        losses = []
        next_element = self.train_data.get_next_batch()
        features = self.sess.run(next_element)

        features = convert_dict_to_list_subdicts(features, self.config.train_batch_size)
        _, _, input_graphs_all_exp, target_graphs_all_exp = create_graphs_and_placeholders(config=self.config, batch_data=features,
                                                                                         batch_size=self.config.train_batch_size)
        for j in range(self.config.train_batch_size):
            loss, _ = self.do_step(input_graphs_all_exp[j], target_graphs_all_exp[j], features[j])
            if loss is not None:
                losses.append(loss)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        batch_loss = np.mean(losses)
        print('step: ', cur_it, ' loss(batch): ', batch_loss)
        summaries_dict = {'loss': batch_loss}
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)

        return batch_loss, cur_it

    def do_step(self, input_graph, target_graphs, feature, train=True):
        exp_length = feature['experiment_length']

        if exp_length != 30:  # todo: remove
            return None, None

        feed_dict = create_feed_dict(self.model.input_ph, self.model.target_ph, input_graph, target_graphs)

        if train:
            data = self.sess.run({"step": self.model.step_op, "target": self.model.target_ph, "loss": self.model.loss_op_train,
                "outputs": self.model.output_ops_train}, feed_dict=feed_dict)

            # print("exp length", exp_length)
            # print("loss", data['loss'])
        else:
            data = self.sess.run({"step": self.model.step_op, "target": self.model.target_ph, "loss": self.model.loss_op_test,
                                  "outputs": self.model.output_ops_test}, feed_dict=feed_dict)

        return data['loss'], data['outputs']

    def test_batch(self):
        losses = []
        next_element = self.test_data.get_next_batch()
        features = self.sess.run(next_element)

        features = convert_dict_to_list_subdicts(features, self.config.test_batch_size)
        _, _, input_graphs_all_exp, target_graphs_all_exp = create_graphs_and_placeholders(config=self.config, batch_data=features,
                                                                                         batch_size=self.config.test_batch_size)
        summaries_dict_rgb = {}
        summaries_dict_seg = {}
        summaries_dict_depth = {}

        target_summaries_dict_rgb = {}

        cur_it = self.model.global_step_tensor.eval(self.sess)
        for i in range(self.config.test_batch_size):
            loss, outputs = self.do_step(input_graphs_all_exp[i], target_graphs_all_exp[i], features[i], train=False)
            if loss is not None:
                losses.append(loss)
            if outputs is not None:
                # returns 3 lists, each having n lists of data lists while n = number of objects
                images_rgb, images_seg, images_depth = get_all_images_from_gn_output(outputs)
                summaries_dict_rgb = {'output_object_{}_img_rgb_{}'.format(j, k): rgb for j, rgb_list in enumerate(images_rgb) for k, rgb in enumerate(rgb_list)}
                summaries_dict_depth = {'output_object_{}_img_depth_{}'.format(j, k): depth for j, depth_list in enumerate(images_depth) for k, depth in enumerate(depth_list)}
                summaries_dict_seg = {'output_object_{}_img_seg_{}'.format(j, k): seg for j, seg_list in enumerate(images_seg) for k, seg in enumerate(seg_list)}
                

                #target_summaries_dict_rgb = {'target_image_seg_object_{}'.format(i): rgb for i, rgb in enumerate(features[j]['img'])}

        batch_loss = np.mean(losses)
        print('step: ', cur_it, ' loss(batch): ', batch_loss)
        summaries_dict = {'loss': batch_loss}
        summaries_dict = {**summaries_dict_rgb, **summaries_dict_seg, **summaries_dict_depth, **summaries_dict}
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        return batch_loss, cur_it


