import numpy as np
import tensorflow as tf

from base.base_train import BaseTrain
from tqdm import tqdm
from utils.utils import convert_dict_to_list_subdicts
from models.singulation_graph import create_graphs_and_placeholders, create_feed_dict


class SingulationTrainer(BaseTrain):
    def __init__(self, sess, model, data, config,logger):
        super(SingulationTrainer, self).__init__(sess, model, data, config, logger)

    def train_epoch(self):
        losses = []

        for i in range(self.data.iterations_per_epoch):
            loss = self.train_step()
            losses.append(loss)

        loss = np.mean(losses)

    def train_step(self):
        ''' one train step = one batch '''
        self.sess.run(self.data.iterator.initializer)
        next_element = self.data.iterator.get_next()
        features = self.sess.run(next_element)

        features = convert_dict_to_list_subdicts(features, self.config.train_batch_size)
        input_phs, target_phs, input_graphs_all_exp, target_graphs_all_exp = create_graphs_and_placeholders(config=self.config,
            batch_data=features, batch_size=self.config.train_batch_size)


        loss = self._process_batch(input_phs, target_phs, input_graphs_all_exp, target_graphs_all_exp, features)
        print(loss)

        summaries_dict = {'loss': loss}
        cur_it = self.model.global_step_tensor.eval(self.sess)
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

        return loss

    def _process_batch(self, input_phs, target_phs, input_graphs_all_exp, target_graphs_all_exp, features):
        batch_loss = 0
        for j in range(self.config.train_batch_size):
            input_ph = input_phs[j]
            target_ph = target_phs[j]
            input_graph = input_graphs_all_exp[j]
            target_graphs = target_graphs_all_exp[j]
            feed_dict = create_feed_dict(input_ph, target_ph, input_graph, target_graphs)

            exp_length = features[j]['experiment_length']
            output_ops_train = self.model(input_ph, exp_length)
            loss_ops_tr = self.model.create_loss_ops(target_ph, output_ops_train)
            loss_op_tr = tf.reduce_sum(loss_ops_tr) / exp_length
            step_op = self.model.optimizer.minimize(loss_op_tr)

            self.sess.run(tf.global_variables_initializer())
            data = self.sess.run({"step": step_op, "target": target_ph, "loss": loss_op_tr, "outputs": output_ops_train},
                                    feed_dict=feed_dict)

            batch_loss += data['loss']

        return np.mean(batch_loss)