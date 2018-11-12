import numpy as np
from base.base_train import BaseTrain
from utils.utils import convert_dict_to_list_subdicts, make_all_runnable_in_session
from models.singulation_graph import create_graphs_and_placeholders, create_feed_dict


class SingulationTrainer(BaseTrain):
    def     __init__(self, sess, model, data, config, logger):
        super(SingulationTrainer, self).__init__(sess, model, data, config, logger)

    def train_epoch(self):
        for i in range(self.data.iterations_per_epoch):
            self.train_step()

    def train_step(self):
        ''' one train step = one batch '''
        next_element = self.data.get_next_batch()
        features = self.sess.run(next_element)

        features = convert_dict_to_list_subdicts(features, self.config.train_batch_size)
        _, _, input_graphs_all_exp, target_graphs_all_exp = create_graphs_and_placeholders(config=self.config,
            batch_data=features, batch_size=self.config.train_batch_size)

        loss = self._process_batch(input_graphs_all_exp, target_graphs_all_exp, features)
        print('batch loss', loss)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        print(cur_it)
        summaries_dict = {'loss': loss}

        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        #self.model.save(self.sess)

        return loss

    def _process_batch(self, input_graphs_all_exp, target_graphs_all_exp, features):
        batch_loss = 0
        for j in range(self.config.train_batch_size):
            input_graph = input_graphs_all_exp[j]
            target_graphs = target_graphs_all_exp[j]
            exp_length = features[j]['experiment_length']

            #if exp_length != 10:
            #    continue

            feed_dict = create_feed_dict(self.model.input_ph, self.model.target_ph, input_graph, target_graphs)
            print(exp_length)


            data = self.sess.run({
                                "step": self.model.step_op,
                                "target": self.model.target_ph,
                                "loss": self.model.loss_op_train,
                                "outputs": self.model.output_ops_train
                                }, feed_dict=feed_dict)

            print(data['loss'])
            batch_loss += data['loss']

        return np.mean(batch_loss)