import numpy as np
import tensorflow as tf
import time
import os
from base.base_train import BaseTrain
from utils.conversions import convert_dict_to_list_subdicts
from utils.tf_summaries import generate_results, generate_and_export_image_dicts
from utils.tf_summaries import generate_results
from utils.io import create_dir
from utils.math_ops import sigmoid
from models.singulation_graph import create_graphs, networkx_graphs_to_images
from joblib import parallel_backend, Parallel, delayed
from eval.compute_test_run_statistics import compute_psnr

class SingulationTrainerNew(BaseTrain):
    def __init__(self, sess, model, train_data, valid_data, config, logger, only_test):
        super(SingulationTrainerNew, self).__init__(sess, model, train_data, valid_data, config, logger, only_test)
        self.next_element_train = self.train_data.get_next_batch()
        self.next_element_test = self.test_data.get_next_batch()

    def train_epoch(self):
        prefix = self.config.exp_name
        print("Running {}".format(prefix))

        while True:
            try:
                cur_batch_it = self.train_batch(prefix)
                if cur_batch_it % self.config.model_save_step_interval == 1:
                    self.model.save(self.sess)
                if cur_batch_it % self.config.test_interval == 1:
                    print("Executing test batch")
                    self.test_batch(prefix, export_images=self.config.export_test_images,
                                   initial_pos_vel_known=self.config.initial_pos_vel_known,
                                    sub_dir_name="tests_during_training")

            except tf.errors.OutOfRangeError:
                break


    def train_batch(self, prefix):
        features = self.sess.run(self.next_element_train)
        features = convert_dict_to_list_subdicts(features, self.config.train_batch_size)

        print("shuffle deactivated")
        input_graphs_batch, target_graphs_batch = create_graphs(config=self.config,
                                                                    batch_data=features,
                                                                    initial_pos_vel_known=self.config.initial_pos_vel_known,
                                                                    batch_processing=True,
                                                                    shuffle=False
                                                                    )
        input_graphs_batch = input_graphs_batch[0]  # todo: delete
        target_graphs_batch = target_graphs_batch[0]  # todo: delete

        in_segxyz, in_image, in_control, gt_label = networkx_graphs_to_images(self.config, input_graphs_batch, target_graphs_batch)

        start_time = time.time()
        last_log_time = start_time


        _, loss_img, out_label = self.sess.run([self.model.train_op, self.model.loss_op, self.out_label_tf], feed_dict={self.in_segxyz_tf: in_segxyz, self.in_image_tf: in_image,
                                                                                                                        self.gt_label_tf: gt_label, self.in_control_tf: in_control,
                                                                                                                        self.is_training: True})
        loss_velocity = np.array(0.0)
        loss_position = np.array(0.0)
        loss_edge = np.array(0.0)
        loss_total = loss_img + loss_position + loss_edge + loss_velocity

        self.sess.run(self.model.increment_cur_batch_tensor)
        cur_batch_it = self.model.cur_batch_tensor.eval(self.sess)

        the_time = time.time()
        elapsed_since_last_log = the_time - last_log_time

        print( 'batch: {:<8} total loss: {:<8.6f} | img loss: {:<8.6f} | vel loss: {:<8.6f} | pos loss: {:<8.6f} | edge loss: {:<8.6f} time(s): {:<10.2f} '
                .format(cur_batch_it, loss_total, loss_img, loss_velocity, loss_position, loss_edge, elapsed_since_last_log)
        )
        summaries_dict = {prefix + '_total_loss': loss_total,
                          prefix + '_img_loss': loss_img,
                          prefix + '_velocity_loss': loss_velocity,
                          prefix + '_position_loss': loss_position,
                          prefix + '_edge_loss': loss_edge
                          }
        self.logger.summarize(cur_batch_it, summaries_dict=summaries_dict, summarizer="train")

        return cur_batch_it


    def test(self):
        if not self.config.n_epochs == 1:
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
                                sub_dir_name="test_{}_iterations_trained".format(cur_batch_it))
            except tf.errors.OutOfRangeError:
                break


    def test_5_objects(self):
        if not self.config.n_epochs == 1:
            print("test mode 5 objects --> n_epochs will be set to 1")
            self.config.n_epochs = 1
        if not self.config.n_rollouts == 50:
            print("test mode 5 objects --> n_rollouts will be set to 50")
            self.config.rollouts = 50
        prefix = self.config.exp_name
        print("Running 5 object test with initial_pos_vel_known={}".format(self.config.initial_pos_vel_known))
        cur_batch_it = self.model.cur_batch_tensor.eval(self.sess)

        if self.config.do_multi_step_prediction:
            sub_dir_name = "test_5_objects_multi_step_{}_iterations_trained".format(cur_batch_it)
        else:
            sub_dir_name = "test_5_novel_object_shapes_single_step_{}_iterations_trained".format(cur_batch_it)

        while True:
            try:
                self.test_batch(prefix=prefix,
                                export_images=self.config.export_test_images,
                                initial_pos_vel_known=self.config.initial_pos_vel_known,
                                process_all_nn_outputs=True,
                                sub_dir_name=sub_dir_name)
            except tf.errors.OutOfRangeError:
                break

    def test_5_objects_multi_step(self):
        # todo: adapt to multi steps
        if not self.config.n_epochs == 1:
            print("test mode 5 objects --> n_epochs will be set to 1")
            self.config.n_epochs = 1
        if not self.config.n_rollouts == 50:
            print("test mode 5 objects --> n_rollouts will be set to 50")
            self.config.rollouts = 50
        prefix = self.config.exp_name
        print("Running 5 object test with initial_pos_vel_known={}".format(self.config.initial_pos_vel_known))
        cur_batch_it = self.model.cur_batch_tensor.eval(self.sess)

        while True:
            try:
                self.test_batch(prefix=prefix,
                                export_images=self.config.export_test_images,
                                initial_pos_vel_known=self.config.initial_pos_vel_known,
                                process_all_nn_outputs=True,
                                sub_dir_name="test_5_objects_multi_step_{}_iterations_trained".format(cur_batch_it))
            except tf.errors.OutOfRangeError:
                break




    def test_batch(self, prefix, initial_pos_vel_known, export_images=False, process_all_nn_outputs=False, sub_dir_name=None,
                   export_latent_data=True, output_results=True):

        losses_total = []
        losses_img = []
        losses_velocity = []
        losses_position = []
        losses_edge = []
        outputs_total = []
        summaries_dict_images = {}

        features = self.sess.run(self.next_element_test)
        features = convert_dict_to_list_subdicts(features, self.config.test_batch_size)

        start_time = time.time()
        last_log_time = start_time

        for i in range(self.config.test_batch_size):
            input_graphs_all_exp, target_graphs_all_exp = create_graphs(config=self.config,
                                                                        batch_data=features[i],
                                                                        initial_pos_vel_known=self.config.initial_pos_vel_known,
                                                                        batch_processing=False,
                                                                        return_only_unpadded=True
                                                                        )
            input_graphs_all_exp = [input_graphs_all_exp]
            target_graphs_all_exp = [target_graphs_all_exp]

            in_segxyz, in_image, in_control, gt_label = networkx_graphs_to_images(self.config, input_graphs_all_exp, target_graphs_all_exp)

            loss_img, out_label = self.sess.run([self.model.loss_op, self.out_label_tf],
                                                   feed_dict={self.in_segxyz_tf: in_segxyz,
                                                              self.in_image_tf: in_image,
                                                              self.gt_label_tf: gt_label,
                                                              self.in_control_tf: in_control,
                                                              self.is_training: False})
            loss_velocity = np.array(0.0)
            loss_position = np.array(0.0)
            loss_edge = np.array(0.0)
            loss_total = loss_img + loss_position + loss_edge + loss_velocity

            losses_total.append(loss_total)
            losses_img.append(loss_img)
            losses_velocity.append(loss_velocity)
            losses_position.append(loss_position)
            losses_edge.append(loss_edge)


            out_label[out_label >= 0.5] = 1.0
            out_label[out_label < 0.5] = 0.0

            outputs_total.append((out_label, in_segxyz, in_image, in_control, i))

        the_time = time.time()
        elapsed_since_last_log = the_time - last_log_time
        cur_batch_it = self.model.cur_batch_tensor.eval(self.sess)

        if not process_all_nn_outputs:
            """ due to brevity, just use last output """
            outputs_total = [outputs_total[-1]]

        batch_loss = np.mean(losses_total)
        img_batch_loss = np.mean(losses_img)
        vel_batch_loss = np.mean(losses_velocity)
        pos_batch_loss = np.mean(losses_position)
        edge_batch_loss = np.mean(losses_edge)

        print('total test batch loss: {:<8.6f} | img loss: {:<8.6f} | vel loss: {:<8.6f} | pos loss {:<8.6f} | edge loss {:<8.6f} time(s): {:<10.2f}'.format(
            batch_loss, img_batch_loss, vel_batch_loss, pos_batch_loss, edge_batch_loss, elapsed_since_last_log))

        summaries_dict = {prefix + '_total_loss': batch_loss,
                          prefix + '_img_loss': img_batch_loss,
                          prefix + '_velocity_loss': vel_batch_loss,
                          prefix + '_position_loss': pos_batch_loss,
                          prefix + '_edge_loss': edge_batch_loss
                          }


        if outputs_total and output_results:
            for output in outputs_total:
                summaries_dict_images = generate_and_export_image_dicts(output=output, features=features, config=self.config,
                                                prefix=prefix, cur_batch_it=cur_batch_it,
                                                dir_name=sub_dir_name, reduce_dict=True)

            if summaries_dict_images:
                summaries_dict = {**summaries_dict, **summaries_dict_images}


        self.logger.summarize(cur_batch_it, summaries_dict=summaries_dict, summarizer="test")

        return batch_loss, img_batch_loss, vel_batch_loss, pos_batch_loss, edge_batch_loss, cur_batch_it

    def test_specific_exp_ids(self):
        assert self.config.n_epochs == 1, "set n_epochs to 1 for test mode"
        prefix = self.config.exp_name
        print("Running tests with initial_pos_vel_known={}".format(self.config.initial_pos_vel_known))
        cur_batch_it = self.model.cur_batch_tensor.eval(self.sess)

        exp_ids_to_export = [13873, 3621, 8575, 439, 2439, 1630, 14526, 4377, 15364, 6874, 11031, 8962]  # big 3 object dataset
        #exp_ids_to_export = [2815, 608, 1691, 49, 922, 1834, 1340, 2596, 2843, 306]  # big 5 object dataset

        export_images = self.config.export_test_images
        export_latent_data = True
        process_all_nn_outputs = True

        thresholds_to_test = [0.4]

        for thresh in thresholds_to_test:
            sub_dir_name = "test_3_objects_specific_exp_ids_{}_iterations_trained_sigmoid_threshold_{}".format(cur_batch_it, thresh)
            while True:
                try:
                    losses_total = []
                    losses_img = []
                    losses_iou = []
                    losses_velocity = []
                    losses_position = []
                    losses_distance = []
                    outputs_total = []

                    features = self.sess.run(self.next_element_test)

                    features = convert_dict_to_list_subdicts(features, self.config.test_batch_size)

                    if exp_ids_to_export:
                        features_to_export = []
                        for dct in features:
                            if dct["experiment_id"] in exp_ids_to_export:
                                features_to_export.append(dct)
                                print("added", dct["experiment_id"])

                        features = features_to_export

                    if exp_ids_to_export and not features_to_export:
                        continue


                    start_time = time.time()
                    last_log_time = start_time

                    for i in range(len(features)):
                        input_graphs_all_exp, target_graphs_all_exp = create_graphs(config=self.config,
                                                                            batch_data=features[i],
                                                                            initial_pos_vel_known=self.config.initial_pos_vel_known,
                                                                            batch_processing=False
                                                                            )
                        output_i = []

                        for j in range(features[i]["unpadded_experiment_length"] - 1):
                            total_loss, output, loss_img, loss_iou, loss_velocity, loss_position, loss_distance = self.do_step(input_graphs_all_exp[j],
                                                                                                           target_graphs_all_exp[j],
                                                                                                           features[i],
                                                                                                           sigmoid_threshold=thresh,
                                                                                                           train=False,
                                                                                                           batch_processing=False
                                                                                                           )
                            output = output[0]
                            if total_loss is not None:
                                losses_total.append(total_loss)
                                losses_img.append(loss_img)
                                losses_iou.append(loss_iou)
                                losses_velocity.append(loss_velocity)
                                losses_position.append(loss_position)
                                losses_distance.append(loss_distance)

                            output_i.append(output)

                        outputs_total.append((output_i, i))

                    the_time = time.time()
                    elapsed_since_last_log = the_time - last_log_time
                    cur_batch_it = self.model.cur_batch_tensor.eval(self.sess)

                    if not process_all_nn_outputs:
                        """ due to brevity, just use last results """
                        outputs_for_summary = [outputs_for_summary[-1]]

                    if losses_total:
                        batch_loss = np.mean(losses_total)
                        img_batch_loss = np.mean(losses_img)
                        iou_batch_loss = np.mean(losses_iou)
                        vel_batch_loss = np.mean(losses_velocity)
                        pos_batch_loss = np.mean(losses_position)
                        dis_batch_loss = np.mean(losses_distance)

                        print('total test batch loss: {:<8.6f} | img loss: {:<10.6f} | iou loss: {:<8.6f} | vel loss: {:<8.6f} | pos loss {:<8.6f} | distance loss {:<8.6f} time(s): {:<10.2f}'.format(
                                batch_loss, img_batch_loss, iou_batch_loss, vel_batch_loss, pos_batch_loss, dis_batch_loss,
                                elapsed_since_last_log))

                    if outputs_total:
                        if self.config.parallel_batch_processing:
                            with parallel_backend('loky', n_jobs=-2):
                                Parallel()(delayed(generate_results)(output, self.config, prefix, features, cur_batch_it, export_images,
                                                              export_latent_data, sub_dir_name) for output in outputs_for_summary)
                        else:
                            for output in outputs_total:
                                generate_results(output=output,
                                                        config=self.config,
                                                        prefix=prefix,
                                                        features=features,
                                                        cur_batch_it=cur_batch_it,
                                                        export_images=export_images,
                                                        export_latent_data=export_latent_data,
                                                        dir_name=sub_dir_name, reduce_dict=True)
                except tf.errors.OutOfRangeError:
                    break
                else:
                    print("continue")
                    continue

