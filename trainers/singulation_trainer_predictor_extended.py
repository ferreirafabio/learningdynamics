import numpy as np
import tensorflow as tf
import time
import os
from base.base_train import BaseTrain
from utils.conversions import convert_dict_to_list_subdicts
from utils.tf_summaries import generate_and_export_image_dicts
from utils.math_ops import compute_iou, compute_precision, compute_recall, compute_f1
from utils.io import create_dir, get_encoding_vectors
from models.singulation_graph import create_graphs, networkx_graphs_to_images
import csv
import pandas as pd


class SingulationTrainerPredictorExtended(BaseTrain):
    def __init__(self, sess, model, train_data, valid_data, config, logger, only_test):
        super(SingulationTrainerPredictorExtended, self).__init__(sess, model, train_data, valid_data, config, logger, only_test)
        self.next_element_train = self.train_data.get_next_batch()
        self.next_element_test = self.test_data.get_next_batch()

        assert os.path.isdir(self.config.perception_features_dir), "{} is not a valid path".format(self.config.perception_features_dir)

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

        if self.config.do_multi_step_prediction:
            multistep = True
        else:
            multistep = False

        start_time = time.time()
        last_log_time = start_time

        input_graphs_batches, target_graphs_batches, random_episode_idx_starts = create_graphs(config=self.config,
                                                                                batch_data=features,
                                                                                initial_pos_vel_known=self.config.initial_pos_vel_known,
                                                                                batch_processing=True,
                                                                                multistep=multistep
                                                                                )

        input_graphs_batches = input_graphs_batches[0]
        target_graphs_batches = target_graphs_batches[0]

        """ gt_label_rec (taken from input graphs) is shifted by -1 compared to gt_label (taken from target graphs) """
        in_segxyz, in_image, in_control, gt_label, gt_label_rec = networkx_graphs_to_images(self.config, input_graphs_batches, target_graphs_batches, multistep=multistep)

        gt_encoder_output, gt_mlp_output = get_encoding_vectors(config=self.config, random_episode_idx_starts=random_episode_idx_starts, train=True)

        if len(gt_encoder_output) == 0 or len( gt_mlp_output) == 0:
            print("at least one .npz file not found, move on")
            cur_batch_it = self.model.cur_batch_tensor.eval(self.sess)
            return cur_batch_it

        gt_latent = np.concatenate([gt_encoder_output, gt_mlp_output])

        _, loss_img, loss_perception, out_predictions, in_rgb_seg_xyz, out_encoder_vectors, dbg_control = self.sess.run([self.model.train_op,
                                                                                                  self.model.img_loss,
                                                                                                  self.model.perception_loss,
                                                                                                  self.out_prediction_softmax,
                                                                                                  self.in_rgb_seg_xyz,
                                                                                                  self.out_latent_vectors,
                                                                                                  self.debug_in_control],
                                                                                                  feed_dict={self.in_segxyz_tf: in_segxyz,
                                                                                                             self.in_image_tf: in_image,
                                                                                                             self.gt_predictions: gt_label,
                                                                                                             self.in_control_tf: in_control,
                                                                                                             self.gt_latent_vectors: gt_latent,
                                                                                                             self.is_training: True})
        loss_velocity = np.array(0.0)
        loss_position = np.array(0.0)
        loss_edge = np.array(0.0)
        loss_total = loss_img + loss_perception + loss_position + loss_edge + loss_velocity

        self.sess.run(self.model.increment_cur_batch_tensor)
        cur_batch_it = self.model.cur_batch_tensor.eval(self.sess)

        the_time = time.time()
        elapsed_since_last_log = the_time - last_log_time

        print( 'batch: {:<8} total loss: {:<8.6f} | img loss: {:<8.6f} | latent loss: {:<8.6f} | vel loss: {:<8.6f} | pos loss: {:<8.6f} | edge loss: {:<8.6f} time(s): {:<10.2f} '
                .format(cur_batch_it, loss_total, loss_img, loss_perception, loss_velocity, loss_position, loss_edge, elapsed_since_last_log)
        )
        summaries_dict = {prefix + '_total_loss': loss_total,
                          prefix + '_img_loss': loss_img,
                          prefix + '_perception_loss': loss_perception,
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

    def test_batch(self, prefix, initial_pos_vel_known, export_images=False, process_all_nn_outputs=False, sub_dir_name=None,
                   export_latent_data=True, output_results=True):

        losses_total = []
        losses_img = []
        losses_perception = []
        losses_velocity = []
        losses_position = []
        losses_edge = []
        outputs_total = []
        summaries_dict_images = {}

        features = self.sess.run(self.next_element_test)
        features = convert_dict_to_list_subdicts(features, self.config.test_batch_size)

        start_time = time.time()
        last_log_time = start_time

        if self.config.do_multi_step_prediction:
            multistep = True
        else:
            multistep = False
        start_idx = 0
        end_idx = self.config.n_predictions

        for i in range(self.config.test_batch_size):
            input_graphs_all_exp, target_graphs_all_exp, random_episode_idx_starts = create_graphs(config=self.config,
                                                                        batch_data=features[i],
                                                                        initial_pos_vel_known=self.config.initial_pos_vel_known,
                                                                        batch_processing=False
                                                                        )

            if multistep:
                input_graphs_all_exp = [input_graphs_all_exp[start_idx:end_idx]]
                target_graphs_all_exp = [target_graphs_all_exp[start_idx:end_idx]]

            gt_encoder_output, gt_mlp_output = get_encoding_vectors(config=self.config, random_episode_idx_starts=random_episode_idx_starts, train=False)

            if len(gt_encoder_output) == 0 or len(gt_mlp_output) == 0:
                print("at least one .npz file not found, move on")
                return

            gt_latent = np.concatenate([gt_encoder_output, gt_mlp_output])

            in_segxyz, in_image, in_control, gt_label, _ = networkx_graphs_to_images(self.config, input_graphs_all_exp, target_graphs_all_exp, multistep=multistep)

            loss_img, loss_perception, out_predictions = self.sess.run([self.model.img_loss,
                                                                        self.model.perception_loss,
                                                                        self.out_prediction_softmax],
                                                                    feed_dict={self.in_segxyz_tf: in_segxyz,
                                                                                  self.in_image_tf: in_image,
                                                                                  self.gt_predictions: gt_label,
                                                                                  self.in_control_tf: in_control,
                                                                                  self.gt_latent_vectors: gt_latent,
                                                                                  self.is_training: False})

            loss_velocity = np.array(0.0)
            loss_position = np.array(0.0)
            loss_edge = np.array(0.0)
            loss_total = loss_img + loss_perception + loss_position + loss_edge + loss_velocity

            losses_total.append(loss_total)
            losses_img.append(loss_img)
            losses_perception.append(loss_perception)
            losses_velocity.append(loss_velocity)
            losses_position.append(loss_position)
            losses_edge.append(loss_edge)

            out_predictions[out_predictions >= 0.5] = 1.0
            out_predictions[out_predictions < 0.5] = 0.0

            outputs_total.append((out_predictions, in_segxyz, in_image, in_control, i, (start_idx, end_idx)))

        the_time = time.time()
        elapsed_since_last_log = the_time - last_log_time
        cur_batch_it = self.model.cur_batch_tensor.eval(self.sess)

        if not process_all_nn_outputs:
            """ due to brevity, just use last output """
            outputs_total = [outputs_total[-1]]

        batch_loss = np.mean(losses_total)
        img_batch_loss = np.mean(losses_img)
        perception_batch_loss = np.mean(losses_perception)
        vel_batch_loss = np.mean(losses_velocity)
        pos_batch_loss = np.mean(losses_position)
        edge_batch_loss = np.mean(losses_edge)

        print('total test batch loss: {:<8.6f} | img loss: {:<8.6f} | latent loss: {:<8.6f} | vel loss: {:<8.6f} | pos loss {:<8.6f} | edge loss {:<8.6f} time(s): {:<10.2f}'.format(
            batch_loss, img_batch_loss, perception_batch_loss, vel_batch_loss, pos_batch_loss, edge_batch_loss, elapsed_since_last_log))

        summaries_dict = {prefix + '_total_loss': batch_loss,
                          prefix + '_img_loss': img_batch_loss,
                          prefix + '_perception_loss': perception_batch_loss,
                          prefix + '_velocity_loss': vel_batch_loss,
                          prefix + '_position_loss': pos_batch_loss,
                          prefix + '_edge_loss': edge_batch_loss
                          }

        if outputs_total and output_results:
            for output in outputs_total:
                summaries_dict_images = generate_and_export_image_dicts(output=output, features=features, config=self.config,
                                                prefix=prefix, cur_batch_it=cur_batch_it,
                                                dir_name=sub_dir_name, reduce_dict=True, multistep=multistep)

            if summaries_dict_images:
                summaries_dict = {**summaries_dict, **summaries_dict_images}

        self.logger.summarize(cur_batch_it, summaries_dict=summaries_dict, summarizer="test")

        return batch_loss, img_batch_loss, vel_batch_loss, pos_batch_loss, edge_batch_loss, cur_batch_it


    def compute_metrics_over_test_set_multistep(self):
        if not self.config.n_epochs == 1:
            print("test mode --> n_epochs will be set to 1")
            self.config.n_epochs = 1
        prefix = self.config.exp_name
        print("Computing IoU, Precision, Recall and F1 score over full test set".format(
            self.config.initial_pos_vel_known))
        cur_batch_it = self.model.cur_batch_tensor.eval(self.sess)

        """ this variable is used to distinguish between 1-step and n-step predictions. Setting it to True or False will call different functions:
         a) when set to False: the function uses "gt_label_rec" as the gt data, inputs are split into episode_length/n_prediction -chunks and fragments 
         (if they cannot be evenly divided) will be dismissed, e.g. episode_length 14 with n_predictions=5 will result in two 5-length chunks
         
         b) when set to True: in this mode, the model only predicts 1-steps. The function uses "in_segxyz" as the ground truth data, 
         the inputs don't have to be processed (e.g. splitting them into episode_length/n_predictions chunks) because after one step, 
         the model is reset to ground truth
          """
        test_single_step = False

        if test_single_step:
            mode_txt = "single_step_tested"
        else:
            mode_txt = "{}_step_tested".format(self.config.n_predictions)

        iou_list_test_set = []
        prec_score_list_test_set = []
        rec_score_list_test_set = []
        f1_score_list_test_set = []
        sub_dir_name = "metric_multistep_models_computation_over_full_test_set_{}_iterations_trained".format(cur_batch_it)

        dir_path, _ = create_dir(os.path.join("../experiments", prefix), sub_dir_name)
        dataset_name = os.path.basename(self.config.tfrecords_dir)
        csv_name = "{}_dataset_{}.csv".format(mode_txt, dataset_name)

        with open(os.path.join(dir_path, csv_name), 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter='\t', lineterminator='\n', )
            writer.writerow(["(metrics averaged over n shapes and full trajectory) mean IoU", "mean precision", "mean recall", "mean f1 over n shapes", "exp_id"])
            while True:
                try:
                    losses_total, losses_img, losses_iou, losses_velocity, losses_position = [], [], [], [], []
                    losses_distance, outputs_total, targets_total, exp_id_total = [], [], [], []

                    features = self.sess.run(self.next_element_test)
                    features = convert_dict_to_list_subdicts(features, self.config.test_batch_size)

                    start_time = time.time()
                    last_log_time = start_time

                    for i in range(self.config.test_batch_size):
                        input_graphs_all_exp, target_graphs_all_exp, _ = create_graphs(config=self.config,
                                                                                    batch_data=features[i],
                                                                                    initial_pos_vel_known=self.config.initial_pos_vel_known,
                                                                                    batch_processing=False,
                                                                                    return_only_unpadded=True,
                                                                                    start_episode=0
                                                                                    )
                        out_label_lst = []
                        in_seg_lst = []
                        exp_id = features[i]['experiment_id']
                        n_objects = features[i]['n_manipulable_objects']

                        if test_single_step:
                            """ create >episode-length< long pairs of input and target as lists to run single prediction steps """
                            assert len(input_graphs_all_exp) == len(target_graphs_all_exp)
                            assert self.config.n_predictions == 1, 'set n_predictions to 1 if single_step is to be tested and re-initialize model'
                            single_step_prediction_chunks_input = [[input_graph] for input_graph in input_graphs_all_exp]
                            single_step_prediction_chunks_target = [[target_graph] for target_graph in target_graphs_all_exp]

                            for lst_inp, lst_targ in zip(single_step_prediction_chunks_input, single_step_prediction_chunks_target):
                                in_segxyz, in_image, in_control, gt_label, _ = networkx_graphs_to_images(self.config,
                                                                                                         [lst_inp],
                                                                                                         [lst_targ],
                                                                                                         multistep=True)

                                gt_latent = np.zeros(shape=(n_objects*self.config.n_predictions, 256))

                                loss_img, out_label = self.sess.run([self.model.img_loss, self.out_prediction_softmax],
                                                                    feed_dict={self.in_segxyz_tf: in_segxyz,
                                                                               self.in_image_tf: in_image,
                                                                               self.gt_predictions: gt_label,
                                                                               self.in_control_tf: in_control,
                                                                               self.gt_latent_vectors: gt_latent,
                                                                               self.is_training: True})

                                out_label[out_label >= 0.5] = 1.0
                                out_label[out_label < 0.5] = 0.0

                                loss_velocity = np.array(0.0)
                                loss_position = np.array(0.0)
                                loss_edge = np.array(0.0)
                                loss_iou = 0.0
                                loss_total = loss_img + loss_position + loss_edge + loss_velocity

                                losses_total.append(loss_total)
                                losses_img.append(loss_img)
                                losses_iou.append(loss_iou)
                                losses_velocity.append(loss_velocity)
                                losses_position.append(loss_position)
                                losses_distance.append(loss_edge)

                                in_seg_lst.append(in_segxyz[:,:,:,0])
                                out_label_lst.append(out_label)

                        else:
                            """ this section is used for producing the output for an entire episode, meaning that the model 
                            is asked to re-predict after n_prediction steps up until the entire episode is covered 
                            (with potential crop in the end if episode length/n_predictions is odd) """
                            assert len(input_graphs_all_exp) == len(target_graphs_all_exp)
                            assert self.config.n_predictions > 1, 'set n_predictions > 1 if multi_step is to be tested and re-initialize model'
                            n_prediction_chunks_target = []
                            n_prediction_chunks_input = []
                            n_predictions = self.config.n_predictions

                            for j in range(0, len(target_graphs_all_exp), n_predictions):
                                chunk = target_graphs_all_exp[j:j + n_predictions]
                                n_prediction_chunks_target.append(chunk)

                                chunk = input_graphs_all_exp[j:j + n_predictions]
                                n_prediction_chunks_input.append(chunk)

                            """ if the length of an episode cannot be evenly divided by n_predictions, remove the last 
                            odd list. end_idx ensures the array split is correctly handled later"""
                            assert all(len(inp_lst) == len(targ_lst) for inp_lst, targ_lst in
                                       zip(n_prediction_chunks_input, n_prediction_chunks_target)), \
                                "input and target lists are not equal after fragmenting them into episode length/n_predictions parts"

                            n_prediction_chunks_target_after_filtering = [chunk for chunk in n_prediction_chunks_target if len(chunk) == n_predictions]
                            n_prediction_chunks_input_after_filtering = [chunk for chunk in n_prediction_chunks_input if len(chunk) == n_predictions]

                            #number_removed_chunks = (len(n_prediction_chunks_target) - len(n_prediction_chunks_target_after_filtering))
                            #print("number of removed chunks (of up to 2 (2 step prediction) or 5 (5 step prediction) frames): ", number_removed_chunks)


                            for lst_inp, lst_targ in zip(n_prediction_chunks_input_after_filtering, n_prediction_chunks_target_after_filtering):
                                in_segxyz, in_image, in_control, gt_label, gt_label_rec = networkx_graphs_to_images(self.config,
                                                                                                         [lst_inp],
                                                                                                         [lst_targ],
                                                                                                         multistep=True)

                                gt_latent = np.zeros(shape=(n_objects * self.config.n_predictions, 256))

                                loss_img, out_label = self.sess.run([self.model.img_loss, self.out_prediction_softmax],
                                                                    feed_dict={self.in_segxyz_tf: in_segxyz,
                                                                               self.in_image_tf: in_image,
                                                                               self.gt_predictions: gt_label,
                                                                               self.in_control_tf: in_control,
                                                                               self.gt_latent_vectors: gt_latent,
                                                                               self.is_training: True})

                                out_label[out_label >= 0.5] = 1.0
                                out_label[out_label < 0.5] = 0.0

                                loss_velocity = np.array(0.0)
                                loss_position = np.array(0.0)
                                loss_edge = np.array(0.0)
                                loss_iou = 0.0
                                loss_total = loss_img + loss_position + loss_edge + loss_velocity

                                losses_total.append(loss_total)
                                losses_img.append(loss_img)
                                losses_iou.append(loss_iou)
                                losses_velocity.append(loss_velocity)
                                losses_position.append(loss_position)
                                losses_distance.append(loss_edge)



                                """ in multistep prediction, in_segxyz only contains the first image, therefore use 
                                gt_label_rec that contains all ground truth input images. Also split the model output 
                                by number of predictions since the model is set-up to append predictions along the 
                                first dimension batch-wise, i.e. ([batch1, batch2]) while both batches each have shape 
                                (5, 120, 160) for 5 objects. This results in total shape (10, 120, 160) and requires a 
                                split for example: 2 predictions, 5 objects: (10, 120, 160) --> [(5, 120, 160), (5, 120, 160)] """
                                gt_label_rec_split = np.split(gt_label_rec, n_predictions, axis=0)
                                out_label_split = np.split(out_label, n_predictions, axis=0)
                                in_seg_lst = in_seg_lst + gt_label_rec_split
                                out_label_lst = out_label_lst + out_label_split

                        out_label_entire_trajectory, in_seg_entire_trajectory = [], []

                        for n in range(n_objects):
                            out_obj_lst = []
                            in_obj_lst = []
                            for time_step_out, time_step_in in zip(out_label_lst, in_seg_lst):
                                out_obj_lst.append(time_step_out[n])
                                in_obj_lst.append(time_step_in[n])
                            out_label_entire_trajectory.append(np.array(out_obj_lst))
                            in_seg_entire_trajectory.append(np.array(in_obj_lst))

                        outputs_total.append(out_label_entire_trajectory)
                        targets_total.append(in_seg_entire_trajectory)
                        exp_id_total.append(exp_id)

                    the_time = time.time()
                    elapsed_since_last_log = the_time - last_log_time
                    batch_loss, img_batch_loss, iou_batch_loss = np.mean(losses_total), np.mean(losses_img), np.mean(losses_iou)
                    vel_batch_loss, pos_batch_loss, dis_batch_loss = np.mean(losses_velocity), np.mean(losses_position), np.mean(losses_distance)
                    print('total test batch loss: {:<8.6f} | img loss: {:<8.6f} | iou loss: {:<8.6f} | vel loss: {:<8.6f} | pos loss {:<8.6f} | edge loss {:<8.6f} time(s): {:<10.2f}'.format(
                            batch_loss, img_batch_loss, iou_batch_loss, vel_batch_loss, pos_batch_loss, dis_batch_loss,
                            elapsed_since_last_log))

                    for pred_experiment, true_experiment, exp_id in zip(outputs_total, targets_total, exp_id_total):
                        iou_scores = []
                        prec_scores = []
                        rec_scores = []
                        f1_scores = []

                        # switch (n_objects, exp_len,...) to (exp_len, n_objects) since IoU computed per time step
                        pred_experiment = np.swapaxes(pred_experiment, 0, 1)
                        true_experiment = np.swapaxes(true_experiment, 0, 1)

                        for pred, true in zip(pred_experiment, true_experiment):
                            iou = compute_iou(pred=pred, true=true)
                            mean_obj_prec_score, idx_obj_min_prec, idx_obj_max_prec = compute_precision(pred=pred, true=true)
                            mean_obj_rec_score, idx_obj_min_rec, idx_obj_max_rec = compute_recall(pred=pred, true=true)
                            mean_obj_f1_score, idx_obj_min_f1, idx_obj_max_f1 = compute_f1(pred=pred, true=true)

                            iou_scores.append(iou)
                            prec_scores.append(mean_obj_prec_score)
                            rec_scores.append(mean_obj_rec_score)
                            f1_scores.append(mean_obj_f1_score)

                        iou_traj_mean = np.mean(iou_scores)
                        prec_traj_mean = np.mean(prec_scores)
                        rec_traj_mean = np.mean(rec_scores)
                        f1_traj_mean = np.mean(f1_scores)

                        if exp_id is None:
                            print("shapes pred_experiment and true experiment: ", np.shape(pred_experiment), np.shape(true_experiment))

                        writer.writerow([iou_traj_mean, prec_traj_mean, rec_traj_mean, f1_traj_mean, exp_id])

                        if not(np.isnan(iou_traj_mean) or np.isnan(prec_traj_mean) or np.isnan(rec_traj_mean) or np.isnan(f1_traj_mean)):
                            prec_score_list_test_set.append(prec_traj_mean)
                            rec_score_list_test_set.append(rec_traj_mean)
                            f1_score_list_test_set.append(f1_traj_mean)
                            iou_list_test_set.append(iou_traj_mean)

                    csv_file.flush()
                except tf.errors.OutOfRangeError:
                    break

            iou_test_set_mean = np.mean(iou_list_test_set)
            prec_test_set_mean = np.mean(prec_score_list_test_set)
            rec_test_set_mean = np.mean(rec_score_list_test_set)
            f1_test_set_mean = np.mean(f1_score_list_test_set)

            writer.writerow(["means over full set", " IoU: ", iou_test_set_mean, " Precision: ", prec_test_set_mean, " Recall: ", rec_test_set_mean, "F1: ", f1_test_set_mean])
            if test_single_step:
                mode = "(model trained multi-step but tested single-step)"
            else:
                mode = "(model trained & tested multi-step)"

            print("Done. mean IoU {}: {}, mean precision: {}, mean recall: {}, mean f1: {}".format(mode, iou_test_set_mean,
                                                                                                prec_test_set_mean,
                                                                                                rec_test_set_mean,
                                                                                                f1_test_set_mean))

    def test_specific_exp_ids(self):
        assert self.config.n_epochs == 1, "test mode for specific exp ids --> n_epochs must be set to 1"
        prefix = self.config.exp_name
        print("Running tests with initial_pos_vel_known={}".format(self.config.initial_pos_vel_known))
        cur_batch_it = self.model.cur_batch_tensor.eval(self.sess)

        if "5_objects_50_rollouts_padded_novel" in self.config.tfrecords_dir:
            exp_ids_to_export = [10, 1206, 880, 1189, 1087, 2261, 194, 1799]  # big 5 object novel shapes dataset
            dir_name = "5_novel_objects"
        elif "5_objects_50_rollouts" in self.config.tfrecords_dir:
            exp_ids_to_export = [2815, 608, 1691, 49, 1834, 1340, 2596, 2843, 306]  # big 5 object dataset
            dir_name = "5_objects"
        else:
            exp_ids_to_export = [13873, 3621, 8575, 439, 2439, 1630, 14526, 4377, 15364, 6874, 11031, 8962]  # big 3 object dataset
            dir_name = "3_objects"

        """ set this to true if full episodes should be predicted with the multistep model, i.e. after every 
        n_predictions, the model input will be reset to the actual ground truth to allow to fully observe the 
        predictions for an entire episode"""
        reset_after_n_predictions = True

        start_idx = 0
        end_idx = self.config.n_predictions

        if self.config.n_predictions > 1 and not reset_after_n_predictions:
            multistep = True
            dir_suffix = "show_pred_from_start"
            start_episode = 0
            pad_ground_truth_to_exp_len = False
        elif self.config.n_predictions > 1 and reset_after_n_predictions:
            multistep = True
            dir_suffix = "reset_model_input_after_{}_predictions".format(self.config.n_predictions)
            start_episode = None
            pad_ground_truth_to_exp_len = False
        else:
            multistep = False
            dir_suffix = ""
            start_episode = 0
            pad_ground_truth_to_exp_len = True

        sub_dir_name = "test_{}_specific_exp_ids_{}_iterations_trained_sigmoid_threshold_{}_mode_{}".format(dir_name, cur_batch_it, 0.5, dir_suffix)
        while True:
            try:
                losses_total = []
                losses_img = []
                losses_velocity = []
                losses_position = []
                losses_edge = []
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
                    input_graphs_all_exp, target_graphs_all_exp, _ = create_graphs(config=self.config,
                                                                                batch_data=features[i],
                                                                                initial_pos_vel_known=self.config.initial_pos_vel_known,
                                                                                batch_processing=False,
                                                                                return_only_unpadded=True,
                                                                                start_episode=start_episode
                                                                                )

                    n_objects = features[i]['n_manipulable_objects']

                    if reset_after_n_predictions:
                        """ this section is used for producing the output for an entire episode, meaning that the model 
                        is asked to re-predict after n_prediction steps up until the entire episode is covered 
                        (with potential crop in the end if episode length/n_predictions is odd) """
                        assert len(input_graphs_all_exp) == len(target_graphs_all_exp)
                        n_prediction_chunks_target = []
                        n_prediction_chunks_input = []
                        n_predictions = self.config.n_predictions

                        for j in range(0, len(target_graphs_all_exp), n_predictions):
                            chunk = target_graphs_all_exp[j:j + n_predictions]
                            n_prediction_chunks_target.append(chunk)

                            chunk = input_graphs_all_exp[j:j + n_predictions]
                            n_prediction_chunks_input.append(chunk)

                        """ if the length of an episode cannot be evenly divided by n_predictions, remove the last 
                        odd list. end_idx ensures the array split is correctly handled later"""
                        n_prediction_chunks_target = [chunk for chunk in n_prediction_chunks_target if len(chunk) == n_predictions]
                        n_prediction_chunks_input = [chunk for chunk in n_prediction_chunks_input if len(chunk) == n_predictions]
                        end_idx = len(n_prediction_chunks_target) * n_predictions

                        out_label_lst = []
                        in_segxyz_lst = []
                        in_image_lst = []
                        in_control_lst = []

                        for lst_inp, lst_targ in zip(n_prediction_chunks_input, n_prediction_chunks_target):
                            in_segxyz, in_image, in_control, gt_label, _ = networkx_graphs_to_images(self.config,
                                                                                                     [lst_inp],
                                                                                                     [lst_targ],
                                                                                                     multistep=True)

                            gt_latent = np.zeros(shape=(n_objects*self.config.n_predictions, 256))

                            loss_img, out_label = self.sess.run([self.model.img_loss, self.out_prediction_softmax],
                                                                feed_dict={self.in_segxyz_tf: in_segxyz,
                                                                           self.in_image_tf: in_image,
                                                                           self.gt_predictions: gt_label,
                                                                           self.in_control_tf: in_control,
                                                                           self.gt_latent_vectors: gt_latent,
                                                                           self.is_training: True})

                            out_label[out_label >= 0.5] = 1.0
                            out_label[out_label < 0.5] = 0.0

                            loss_velocity = np.array(0.0)
                            loss_position = np.array(0.0)
                            loss_edge = np.array(0.0)
                            loss_total = loss_img + loss_position + loss_edge + loss_velocity

                            losses_total.append(loss_total)
                            losses_img.append(loss_img)
                            losses_velocity.append(loss_velocity)
                            losses_position.append(loss_position)
                            losses_edge.append(loss_edge)

                            out_label_lst.append(out_label)
                            in_segxyz_lst.append(in_segxyz)
                            in_image_lst.append(in_image)
                            in_control_lst.append(in_control)

                        out_label, in_segxyz, in_image, in_control = np.concatenate(out_label_lst, axis=0), \
                                                                     np.concatenate(in_segxyz_lst, axis=0), \
                                                                     np.concatenate(in_image_lst, axis=0), \
                                                                     np.concatenate(in_control_lst, axis=0)

                        outputs_total.append((out_label, in_segxyz, in_image, in_control, i, (start_idx, end_idx)))


                    else:
                        if multistep:
                            input_graphs_all_exp = [input_graphs_all_exp[start_idx:end_idx]]
                            target_graphs_all_exp = [target_graphs_all_exp[start_idx:end_idx]]

                        in_segxyz, in_image, in_control, gt_label, _ = networkx_graphs_to_images(self.config,
                                                                                                 input_graphs_all_exp,
                                                                                                 target_graphs_all_exp,
                                                                                                 multistep=multistep)

                        gt_latent = np.zeros(shape=(n_objects, 256))

                        loss_img, out_label = self.sess.run([self.model.img_loss, self.out_prediction_softmax],
                                                            feed_dict={self.in_segxyz_tf: in_segxyz,
                                                                       self.in_image_tf: in_image,
                                                                       self.gt_predictions: gt_label,
                                                                       self.in_control_tf: in_control,
                                                                       self.gt_latent_vectors: gt_latent,
                                                                       self.is_training: True})
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

                        outputs_total.append((out_label, in_segxyz, in_image, in_control, i, (start_idx, end_idx)))

                the_time = time.time()
                elapsed_since_last_log = the_time - last_log_time
                cur_batch_it = self.model.cur_batch_tensor.eval(self.sess)

                batch_loss = np.mean(losses_total)
                img_batch_loss = np.mean(losses_img)
                vel_batch_loss = np.mean(losses_velocity)
                pos_batch_loss = np.mean(losses_position)
                edge_batch_loss = np.mean(losses_edge)

                print(
                    'total test batch loss: {:<8.6f} | img loss: {:<8.6f} | vel loss: {:<8.6f} | pos loss {:<8.6f} | edge loss {:<8.6f} time(s): {:<10.2f}'.format(
                        batch_loss, img_batch_loss, vel_batch_loss, pos_batch_loss, edge_batch_loss,
                        elapsed_since_last_log))

                if outputs_total:
                    for output in outputs_total:
                        generate_and_export_image_dicts(output=output, features=features,
                                                                    config=self.config,
                                                                    prefix=prefix,
                                                                    cur_batch_it=cur_batch_it,
                                                                    dir_name=sub_dir_name,
                                                                    reduce_dict=True,
                                                                    multistep=multistep,
                                                                    pad_ground_truth_to_exp_len=pad_ground_truth_to_exp_len)

            except tf.errors.OutOfRangeError:
                break
            else:
                print("continue")
                continue

    def store_latent_vectors(self):
        assert self.config.n_epochs == 1, "set n_epochs to 1 for test mode"
        assert self.config.test_batch_size == 1, "set test_batch_size to 1 for test mode"
        prefix = self.config.exp_name
        print("Storing latent vectors baseline")
        cur_batch_it = self.model.cur_batch_tensor.eval(self.sess)

        df = pd.DataFrame(columns=['latent_vector_init_img', 'exp_id', 'exp_len'])
        sub_dir_name = "latent_vectors_initial_image_of_full_test_set_{}_iterations_trained".format(cur_batch_it)

        dir_path, _ = create_dir(os.path.join("../experiments", prefix), sub_dir_name)
        dataset_name = os.path.basename(self.config.tfrecords_dir)
        file_name = dir_path + "/latent_vectors_baseline_auto_predictor_dataset_{}.pkl".format(dataset_name)

        while True:
            try:
                features = self.sess.run(self.next_element_test)
                features = convert_dict_to_list_subdicts(features, self.config.test_batch_size)

                for i in range(len(features)):
                    input_graphs_all_exp, target_graphs_all_exp, _ = create_graphs(config=self.config, batch_data=features[i],
                                                                                initial_pos_vel_known=self.config.initial_pos_vel_known,
                                                                                batch_processing=False)

                    input_graphs_all_exp = [input_graphs_all_exp[0]]
                    target_graphs_all_exp = [target_graphs_all_exp[0]]

                    exp_id = features[i]['experiment_id']
                    exp_len = features[i]["unpadded_experiment_length"]  # the label

                    #print(np.shape(input_graphs_all_exp), np.shape(target_graphs_all_exp))

                    input_graphs_all_exp = [input_graphs_all_exp]
                    target_graphs_all_exp = [target_graphs_all_exp]

                    in_segxyz, in_image, in_control, gt_label = networkx_graphs_to_images(self.config, input_graphs_all_exp,
                                                                                          target_graphs_all_exp)

                    #print(np.shape(in_segxyz), np.shape(in_image), np.shape(in_control), np.shape(gt_label), exp_len)

                    loss_img, out_label, latent_init_img = self.sess.run([self.model.loss_op, self.out_prediction_softmax, self.latent_init_img],
                                                                         feed_dict={self.in_segxyz_tf: in_segxyz, self.in_image_tf: in_image,
                                                                   self.gt_predictions: gt_label, self.in_control_tf: in_control,
                                                                   self.is_training: True})

                    #print(np.shape(latent_init_img))

                    df = df.append({'latent_vector_init_img': latent_init_img, 'exp_id': exp_id, 'exp_len': exp_len}, ignore_index=True)

            except tf.errors.OutOfRangeError:
                df.to_pickle(file_name)
                print("Pandas dataframe with {} rows saved to: {} ".format(len(df.index), file_name))
                break
            else:
                print("continue")
                continue