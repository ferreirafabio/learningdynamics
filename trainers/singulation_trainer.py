import numpy as np
import tensorflow as tf
from base.base_train import BaseTrain
from utils.utils import convert_dict_to_list_subdicts, make_all_runnable_in_session
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
            loss = self.do_step(input_graphs_all_exp[j], target_graphs_all_exp[j], features[j])
            if loss is not None:
                losses.append(loss)

        batch_loss = np.mean(losses)
        cur_it = self.model.global_step_tensor.eval(self.sess)
        print('step: ', cur_it, ' loss(batch): ', batch_loss)
        summaries_dict = {'loss': batch_loss}
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)

        return batch_loss, cur_it

    def do_step(self, input_graph, target_graphs, feature, train=True):
        exp_length = feature['experiment_length']

        if exp_length != 30:  # todo: remove
            return

        feed_dict = create_feed_dict(self.model.input_ph, self.model.target_ph, input_graph, target_graphs)

        if train:
            data = self.sess.run({"step": self.model.step_op, "target": self.model.target_ph, "loss": self.model.loss_op_train,
                "outputs": self.model.output_ops_train}, feed_dict=feed_dict)

            # print("exp length", exp_length)
            # print("loss", data['loss'])
        else:
            data = self.sess.run({"step": self.model.step_op, "target": self.model.target_ph, "loss": self.model.loss_op_test,
                                  "outputs": self.model.output_ops_test}, feed_dict=feed_dict)

        return data['loss']

    def test_batch(self):
        losses = []
        next_element = self.test_data.get_next_batch()
        features = self.sess.run(next_element)

        features = convert_dict_to_list_subdicts(features, self.config.test_batch_size)
        _, _, input_graphs_all_exp, target_graphs_all_exp = create_graphs_and_placeholders(config=self.config, batch_data=features,
                                                                                         batch_size=self.config.test_batch_size)

        for j in range(self.config.test_batch_size):
            loss = self.do_step(input_graphs_all_exp[j], target_graphs_all_exp[j], features[j], train=False)
            if loss is not None:
                losses.append(loss)

        batch_loss = np.mean(losses)
        cur_it = self.model.global_step_tensor.eval(self.sess)
        print('step: ', cur_it, ' loss(batch): ', batch_loss)
        summaries_dict = {'loss': batch_loss}
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)

        return batch_loss, cur_it
