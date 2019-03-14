import numpy as np
import tensorflow as tf
import time
from base.base_train import BaseTrain
from utils.conversions import convert_dict_to_list_subdicts, denormalize_gn_output
from utils.tf_summaries import generate_results
from models.singulation_graph import create_graphs, create_feed_dict
from joblib import parallel_backend, Parallel, delayed

class SingulationTrainer(BaseTrain):
    def __init__(self, sess, model, train_data, valid_data, config, logger):
        super(SingulationTrainer, self).__init__(sess, model, train_data, valid_data, config, logger)

    def train_epoch(self):
        prefix = self.config.exp_name
        while True:
            try:
                _, _, _, _, cur_batch_it = self.train_batch(prefix)
                if cur_batch_it % self.config.model_save_step_interval == 1:
                    self.model.save(self.sess)
                if cur_batch_it % self.config.test_interval == 1:
                    print("Executing test batch")
                    self.test_batch(prefix, export_images=self.config.export_test_images,
                                    initial_pos_vel_known=self.config.initial_pos_vel_known,
                                    sub_dir_name="tests_during_training")

            except tf.errors.OutOfRangeError:
                break

    def do_step(self, input_graph, target_graphs, feature, train=True):

        if train:
            feed_dict = create_feed_dict(self.model.input_ph, self.model.target_ph, input_graph, target_graphs)
            data = self.sess.run({"step": self.model.step_op,
                                  "target": self.model.target_ph,
                                  "loss_total": self.model.loss_op_train_total,
                                  "outputs": self.model.output_ops_train,
                                  "loss_img": self.model.loss_ops_train_img,
                                  "loss_velocity": self.model.loss_ops_train_velocity,
                                  "loss_position": self.model.loss_ops_train_position,
                                  "loss_distance": self.model.loss_ops_train_distance
                                  }, feed_dict=feed_dict)

        else:

            feed_dict = create_feed_dict(self.model.input_ph_test, self.model.target_ph_test, input_graph, target_graphs)
            data = self.sess.run({"target": self.model.target_ph_test,
                                  "loss_total": self.model.loss_op_test_total,
                                  "outputs": self.model.output_ops_test,
                                  "loss_img": self.model.loss_ops_test_img,
                                  "loss_velocity": self.model.loss_ops_test_velocity,
                                  "loss_position": self.model.loss_ops_test_position,
                                  "loss_distance": self.model.loss_ops_test_distance
                                  }, feed_dict=feed_dict)

        return data['loss_total'], data['outputs'], data['loss_img'], data['loss_velocity'], data['loss_position'], data['loss_distance']

    def test_rollouts(self):
        if self.config.n_epochs == 1:
            print("test mode --> n_epochs will be set to 1")
            self.config.n_epochs = 1
        prefix = self.config.exp_name
        print("Running tests with initial_pos_vel_known={}".format(self.config.initial_pos_vel_known))
        cur_batch_it = self.model.cur_batch_tensor.eval(self.sess)

        while True:
            try:
                self.test_batch(prefix=prefix,
                                export_images=self.config.export_test_images,
                                initial_pos_vel_known=self.config.initial_pos_vel_known,
                                process_all_nn_outputs=True,
                                sub_dir_name="test_{}_rollouts_{}_iterations_trained".format(self.config.n_rollouts, cur_batch_it))
            except tf.errors.OutOfRangeError:
                break

    def test_overfit(self):
        if self.config.n_epochs == 1:
            print("test mode --> n_epochs will be set to 1")
            self.config.n_epochs = 1
        prefix = self.config.exp_name
        print("Running tests with initial_pos_vel_known={}".format(self.config.initial_pos_vel_known))
        cur_batch_it = self.model.cur_batch_tensor.eval(self.sess)

        while True:
            try:
                self.test_batch(prefix=prefix,
                                export_images=self.config.export_test_images,
                                initial_pos_vel_known=self.config.initial_pos_vel_known,
                                process_all_nn_outputs=True,
                                sub_dir_name="test_overfit_{}_rollouts_{}_iterations_trained".format(self.config.n_rollouts, cur_batch_it))
            except tf.errors.OutOfRangeError:
                break


    def train_batch(self, prefix):
        losses = []
        losses_img = []
        losses_velocity = []
        losses_position = []
        losses_distance = []

        next_element = self.train_data.get_next_batch()
        features = self.sess.run(next_element)

        features = convert_dict_to_list_subdicts(features, self.config.train_batch_size)
        input_graphs_all_exp, target_graphs_all_exp = create_graphs(config=self.config,
                                                                    batch_data=features,
                                                                    batch_size=self.config.train_batch_size,
                                                                    initial_pos_vel_known=self.config.initial_pos_vel_known)

        start_time = time.time()
        last_log_time = start_time

        if self.config.parallel_batch_processing:
            with parallel_backend('threading', n_jobs=-2):
                losses, pos_vel_losses = Parallel()(delayed(self._do_step_parallel)(input_graphs_all_exp[i], target_graphs_all_exp[i],
                                                                                    features[i], losses) for i in range(self.config.train_batch_size))
        else:
            for i in range(self.config.train_batch_size):
                total_loss, _, loss_img, loss_velocity, loss_position, loss_distance = self.do_step(input_graphs_all_exp[i],
                                                                                                    target_graphs_all_exp[i], features[i])
                if total_loss is not None:
                    losses.append(total_loss)
                    losses_img.append(loss_img)
                    losses_velocity.append(loss_velocity)
                    losses_position.append(loss_position)
                    losses_distance.append(loss_distance)

        the_time = time.time()
        elapsed_since_last_log = the_time - last_log_time

        self.sess.run(self.model.increment_cur_batch_tensor)
        cur_batch_it = self.model.cur_batch_tensor.eval(self.sess)

        if losses:
            batch_loss = np.mean(losses)
            img_batch_loss = np.mean(losses_img)
            vel_batch_loss = np.mean(losses_velocity)
            pos_batch_loss = np.mean(losses_position)
            dis_batch_loss = np.mean(losses_distance)

            print('batch: {:<8} total loss: {:<8.6f} | img loss: {:<8.6f} | vel loss: {:<8.6f} | pos loss: {:<8.6f} | distance loss: {:<8.6f} time(s): {:<10.2f} '
                .format(
                cur_batch_it, batch_loss, img_batch_loss, vel_batch_loss, pos_batch_loss, dis_batch_loss, elapsed_since_last_log)
            )
            summaries_dict = {prefix + '_total_loss': batch_loss,
                              prefix + '_img_loss': img_batch_loss,
                              prefix + '_velocity_loss': vel_batch_loss,
                              prefix + '_position_loss': pos_batch_loss,
                              prefix + '_distance_loss': dis_batch_loss
                              }
            self.logger.summarize(cur_batch_it, summaries_dict=summaries_dict, summarizer="train")
        else:
            batch_loss, img_batch_loss, vel_batch_loss, pos_batch_loss, dis_batch_loss = 0, 0, 0, 0, 0

        return batch_loss, vel_batch_loss, pos_batch_loss, dis_batch_loss, cur_batch_it

    def test_batch(self, prefix, initial_pos_vel_known, export_images=False, process_all_nn_outputs=False, sub_dir_name=None, export_latent_data=True, add_gripper_noise=False):
        losses_total = []
        losses_img = []
        losses_velocity = []
        losses_position = []
        losses_distance = []
        outputs_for_summary = []
        summaries_dict = {}
        summaries_dict_images = {}

        next_element = self.test_data.get_next_batch()
        features = self.sess.run(next_element)



        features = convert_dict_to_list_subdicts(features, self.config.test_batch_size)
        #if add_gripper_noise:


        input_graphs_all_exp, target_graphs_all_exp = create_graphs(config=self.config,
                                                                    batch_data=features,
                                                                    batch_size=self.config.test_batch_size,
                                                                    initial_pos_vel_known=initial_pos_vel_known
                                                                    )


        start_time = time.time()
        last_log_time = start_time

        for i in range(self.config.test_batch_size):
            total_loss, outputs, loss_img, loss_velocity, loss_position, loss_distance = self.do_step(input_graphs_all_exp[i],
                                                                                                      target_graphs_all_exp[i],
                                                                                                      features[i],
                                                                                                      train=False)
            if total_loss is not None:
                losses_total.append(total_loss)
                losses_img.append(loss_img)
                losses_velocity.append(loss_velocity)
                losses_position.append(loss_position)
                losses_distance.append(loss_distance)

            ''' get the last not-None output '''
            if outputs is not None:
                outputs_for_summary.append((outputs, i))

        the_time = time.time()
        elapsed_since_last_log = the_time - last_log_time
        cur_batch_it = self.model.cur_batch_tensor.eval(self.sess)

        if not process_all_nn_outputs:
            """ due to brevity, just use last results """
            outputs_for_summary = [outputs_for_summary[-1]]

        if total_loss:
            batch_loss = np.mean(losses_total)
            img_batch_loss = np.mean(losses_img)
            vel_batch_loss = np.mean(losses_velocity)
            pos_batch_loss = np.mean(losses_position)
            dis_batch_loss = np.mean(losses_distance)


            print('total test batch loss: {:<8.6f} | img loss: {:<8.6f} | vel loss: {:<8.6f} | pos loss {:<8.6f} | distance loss {:<8.6f} time(s): {:<10.2f}'.format(
                batch_loss, img_batch_loss, vel_batch_loss, pos_batch_loss, dis_batch_loss, elapsed_since_last_log))

            summaries_dict = {prefix + '_total_loss': batch_loss,
                              prefix + '_img_loss': img_batch_loss,
                              prefix + '_velocity_loss': vel_batch_loss,
                              prefix + '_position_loss': pos_batch_loss,
                              prefix + '_distance_loss': dis_batch_loss
                              }

        else:
            batch_loss, img_batch_loss, vel_batch_loss, pos_batch_loss, dis_batch_loss = 0, 0, 0, 0, 0

        if outputs_for_summary is not None:
            if self.config.parallel_batch_processing:
                with parallel_backend('loky', n_jobs=-1):
                    summaries_dict_images, summaries_pos_dict_images = Parallel()(delayed(generate_results)(output, self.config, prefix, features, cur_batch_it,
                                                                                 export_images, export_latent_data, sub_dir_name) for output in outputs_for_summary)


            else:
                for output in outputs_for_summary:
                    summaries_dict_images, summaries_pos_dict_images = generate_results(output=output,
                                                                     config=self.config,
                                                                     prefix=prefix,
                                                                     features=features,
                                                                     cur_batch_it=cur_batch_it,
                                                                     export_images=export_images,
                                                                     export_latent_data=export_latent_data,
                                                                     dir_name=sub_dir_name)

            if summaries_dict_images:
                if self.config.parallel_batch_processing:
                    """ parallel mode returns list, just use first element as a summary for the logger """
                    summaries_dict_images = summaries_dict_images[0]
                    if summaries_pos_dict_images is not None: 
                        summaries_pos_dict_images = summaries_pos_dict_images[0]
                
                if summaries_pos_dict_images is not None:
                    # todo: add pos_dict
                    summaries_dict = {**summaries_dict, **summaries_dict_images}
                else:
                    summaries_dict = {**summaries_dict, **summaries_dict_images}
                cur_batch_it = self.model.cur_batch_tensor.eval(self.sess)
                self.logger.summarize(cur_batch_it, summaries_dict=summaries_dict, summarizer="test")

        return batch_loss, vel_batch_loss, pos_batch_loss, dis_batch_loss, cur_batch_it




    def _do_step_parallel(self, input_graphs_all_exp, target_graphs_all_exp, features, losses, pos_vel_losses):
        loss, _, pos_vel_loss = self.do_step(input_graphs_all_exp, target_graphs_all_exp, features, train=True)
        if loss is not None:
            losses.append(loss)
            pos_vel_losses.append(pos_vel_loss)
        return losses, pos_vel_loss


