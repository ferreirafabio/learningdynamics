import numpy as np
import tensorflow as tf
import time
import os
import csv
from base.base_train import BaseTrain
from utils.conversions import convert_dict_to_list_subdicts
from utils.tf_summaries import generate_results
from utils.io import create_dir
from utils.math_ops import sigmoid, compute_iou, compute_precision, compute_recall, compute_f1
from utils.utils import extract_input_and_output
from models.singulation_graph import create_graphs, create_feed_dict
from joblib import parallel_backend, Parallel, delayed
from eval.compute_test_run_statistics import compute_psnr

class SingulationTrainer(BaseTrain):
    def __init__(self, sess, model, train_data, valid_data, config, logger, only_test):
        super(SingulationTrainer, self).__init__(sess, model, train_data, valid_data, config, logger, only_test=only_test)
        self.next_element_train = self.train_data.get_next_batch()
        self.next_element_test = self.test_data.get_next_batch()

    def train_epoch(self):
        prefix = self.config.exp_name
        print("rollout steps set to {}".format(self.config.n_rollouts))
        n_batches_trained_since_last_test = 0
        n_batches_trained_since_last_save = 0
        while True:
            try:
                if self.config.batch_processing:
                    cur_batch_it, n_batches_trained_since_last_test, n_batches_trained_since_last_save = \
                        self.train_multiple_batches(prefix, n_batches_trained_since_last_test, n_batches_trained_since_last_save)

                    if n_batches_trained_since_last_test > self.config.test_interval:
                        n_batches_trained_since_last_test = 0
                        print("Executing test batch")
                        self.test_batch(prefix, export_images=self.config.export_test_images,
                                        initial_pos_vel_known=self.config.initial_pos_vel_known,
                                        sub_dir_name="tests_during_training")

                    if n_batches_trained_since_last_save > self.config.model_save_step_interval:
                        n_batches_trained_since_last_save = 0
                        self.model.save(self.sess)

                else:
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

    def do_step(self, input_graph, target_graphs, feature, train=True, sigmoid_threshold=0.4, batch_processing=True):

        if train:
            self.model.input_ph, self.model.target_ph, feed_dict = create_feed_dict(self.model.input_ph, self.model.target_ph,
                                         input_graph, target_graphs, self.config, batch_processing=batch_processing)

            data = self.sess.run({"step": self.model.step_op,
                                  "target": self.model.target_ph,
                                  #"logits": self.model.train_logits,
                                  "loss_total": self.model.loss_op_train_total,
                                  "outputs": self.model.output_ops_train,
                                  "loss_img": self.model.loss_ops_train_img,
                                  "loss_iou": self.model.loss_ops_train_iou,
                                  "loss_velocity": self.model.loss_ops_train_velocity,
                                  "loss_position": self.model.loss_ops_train_position,
                                  "loss_distance": self.model.loss_ops_train_distance,
                                  }, feed_dict=feed_dict)

        else:
            self.model.input_ph_test, self.model.target_ph_test, feed_dict = create_feed_dict(self.model.input_ph_test, self.model.target_ph_test,
                                         input_graph, target_graphs, self.config, batch_processing=batch_processing)

            data = self.sess.run({"target": self.model.target_ph_test,
                                  "loss_total": self.model.loss_op_test_total,
                                  #"logits": self.model.test_logits,
                                  "outputs": self.model.output_ops_test,
                                  "loss_img": self.model.loss_ops_test_img,
                                  "loss_iou": self.model.loss_ops_test_iou,
                                  "loss_velocity": self.model.loss_ops_test_velocity,
                                  "loss_position": self.model.loss_ops_test_position,
                                  "loss_distance": self.model.loss_ops_test_distance,
                                  }, feed_dict=feed_dict)


            """ currently we are interested in values in {0,1} since we train binary cross entropy (segmentation images). 
            tf.nn.sigmoid_cross_entropy_with_logits runs the logits through sigmoid() which is why for outputs we also 
            need to run a sigmoid(). """
            for output in data['outputs']:
                seg_data = output.nodes[:, :-6]
                seg_data = np.reshape(seg_data, [-1, 120, 160, 2])
                seg_data = sigmoid(seg_data)
                seg_data = seg_data[:, :, :, 1]  # always select the 2nd feature map containing the 1's
                seg_data = np.reshape(seg_data, [-1, 19200])
                seg_data[seg_data >= sigmoid_threshold] = 1.0
                seg_data[seg_data < sigmoid_threshold] = 0.0
                output.nodes[:, :-6-19200] = seg_data
        return data['loss_total'], data['outputs'], data['loss_img'], data['loss_iou'], data['loss_velocity'], data['loss_position'], data['loss_distance'], data['target']


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

    def compute_losses_over_test_set(self):
        if not self.config.n_epochs == 1:
            print("test mode --> n_epochs will be set to 1")
            self.config.n_epochs = 1
        prefix = self.config.exp_name
        print("Computing average losses over full test set with initial_pos_vel_known={}".format(self.config.initial_pos_vel_known))
        cur_batch_it = self.model.cur_batch_tensor.eval(self.sess)

        full_batch_loss, full_img_loss, full_vel_loss, full_pos_loss, full_dist_loss, full_iou_loss = [], [], [], [], [], []

        while True:
            try:
                batch_total, img_loss, vel_loss, pos_loss, dist_loss, iou_loss, _, _, _, _ = self.test_batch(prefix=prefix,
                                                                    export_images=self.config.export_test_images,
                                                                    initial_pos_vel_known=self.config.initial_pos_vel_known,
                                                                    process_all_nn_outputs=True,
                                                                    sub_dir_name="test_{}_iterations_trained".format(cur_batch_it),
                                                                    output_results=False)
                full_batch_loss.append(batch_total)
                full_img_loss.append(img_loss)
                full_vel_loss.append(vel_loss)
                full_pos_loss.append(pos_loss)
                full_dist_loss.append(dist_loss)
                full_iou_loss.append(iou_loss)

            except tf.errors.OutOfRangeError:
                break
        mean_total_loss = np.mean(full_batch_loss)
        mean_img_loss = np.mean(full_img_loss)
        mean_vel_loss = np.mean(full_vel_loss)
        mean_pos_loss = np.mean(full_pos_loss)
        mean_dist_loss = np.mean(full_dist_loss)
        mean_iou_loss = np.mean(full_iou_loss)

        output_dir, _ = create_dir(os.path.join("../experiments", prefix), "loss_over_test_set")
        dataset_name = os.path.basename(self.config.tfrecords_dir)

        if self.config.loss_type == "cross_entropy_seg_only":
            img_loss_type = "mean binary cross entropy loss"
        else:
            img_loss_type = "mean img loss"

        str_out = "mean loss over all test samples of dataset: {}\nmean total loss: " \
                  "{}\n{}: {}\nmean vel loss: {}\nmean pos loss: {}\nmean dist loss: {}\nmean iou loss: {}".\
            format(self.config.tfrecords_dir, mean_total_loss, img_loss_type, mean_img_loss, mean_vel_loss, mean_pos_loss, mean_dist_loss, mean_iou_loss)

        with open(output_dir + '/mean_losses_over_full_test_set_of_{}.txt'.format(dataset_name), "a+") as text_file:
            text_file.write(str_out + "\n")

        print(str_out)

        return mean_total_loss, mean_vel_loss, mean_pos_loss, mean_dist_loss, mean_iou_loss

    def compute_iou_over_test_set(self):
        if not self.config.n_epochs == 1:
            print("test mode --> n_epochs will be set to 1")
            self.config.n_epochs = 1
        prefix = self.config.exp_name
        print("Computing IoU over full test set with initial_pos_vel_known={}".format(
            self.config.initial_pos_vel_known))
        cur_batch_it = self.model.cur_batch_tensor.eval(self.sess)

        iou_list_test_set = []
        prec_score_list_test_set = []
        rec_score_list_test_set = []
        f1_score_list_test_set = []
        full_batch_loss, full_img_loss, full_vel_loss, full_pos_loss, full_dist_loss, full_iou_loss = [], [], [], [], [], []
        sub_dir_name = "iou_metric_over_full_3_objects_test_set_{}_iterations_trained".format(cur_batch_it)
        dir_path, _ = create_dir(os.path.join("../experiments", prefix), sub_dir_name)
        with open(os.path.join(dir_path, 'iou_test_set.csv'), 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter='\t', lineterminator='\n', )
            writer.writerow(["mean IoU over n shapes", "exp_id", "mean precision over n shapes", "mean recall over n shapes", "mean f1 over n shapes"])
            while True:
                try:


                    batch_total, img_loss, vel_loss, pos_loss, dist_loss, iou_loss, _, outputs, targets, exp_ids = self.test_batch(prefix=prefix,
                                                                        export_images=False,
                                                                        initial_pos_vel_known=self.config.initial_pos_vel_known,
                                                                        process_all_nn_outputs=True,
                                                                        sub_dir_name="test_{}_iterations_trained".format(cur_batch_it),
                                                                        output_results=False)

                    predictions_list, ground_truth_list = extract_input_and_output(outputs=outputs, targets=targets)
                    exp_ids = [exp_id_sub for exp_id_sup in exp_ids for exp_id_sub in exp_id_sup]
                    for pred, true, exp_id in zip(predictions_list, ground_truth_list, exp_ids):

                        iou = compute_iou(pred=pred, true=true)
                        mean_obj_prec_score, idx_obj_min_prec, idx_obj_max_prec = compute_precision(pred=pred, true=true)
                        mean_obj_rec_score, idx_obj_min_rec, idx_obj_max_rec = compute_recall(pred=pred, true=true)
                        mean_obj_f1_score, idx_obj_min_f1, idx_obj_max_f1 = compute_f1(pred=pred, true=true)

                        prec_score_list_test_set.append(mean_obj_prec_score)
                        rec_score_list_test_set.append(mean_obj_rec_score)
                        f1_score_list_test_set.append(mean_obj_f1_score)
                        iou_list_test_set.append(iou)

                        writer.writerow([iou, exp_id, mean_obj_prec_score, mean_obj_rec_score, mean_obj_f1_score])

                    csv_file.flush()
                except tf.errors.OutOfRangeError:
                    break

            iou_mean = np.mean(iou_list_test_set)
            prec_mean = np.mean(prec_score_list_test_set)
            rec_mean = np.mean(rec_score_list_test_set)
            f1_mean = np.mean(f1_score_list_test_set)

            writer.writerow(["means over full set", " IoU: ", iou_mean, " Precision: ", prec_mean, " Recall: ", rec_mean, "F1: ", f1_mean])

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

    def train_batch(self, prefix):
        losses = []
        losses_img = []
        losses_iou = []
        losses_velocity = []
        losses_position = []
        losses_distance = []

        features = self.sess.run(self.next_element_train)

        features = convert_dict_to_list_subdicts(features, self.config.train_batch_size)

        start_time = time.time()
        last_log_time = start_time


        for i in range(self.config.train_batch_size):
            input_graphs_all_exp, target_graphs_all_exp = create_graphs(config=self.config,
                                                                        batch_data=features[i],
                                                                        initial_pos_vel_known=self.config.initial_pos_vel_known,
                                                                        batch_processing=False
                                                                        )


            for j in range(features[i]["unpadded_experiment_length"]-1):
                total_loss, _, loss_img, loss_iou, loss_velocity, loss_position, loss_distance, _ = self.do_step(input_graphs_all_exp[j],
                                                                                                       target_graphs_all_exp[j],
                                                                                                       features[i],
                                                                                                       train=True,
                                                                                                       batch_processing=False
                                                                                                       )

                if total_loss is not None:
                    losses.append(total_loss)
                    losses_img.append(loss_img)
                    losses_iou.append(loss_iou)
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
            iou_batch_loss = np.mean(losses_iou)
            vel_batch_loss = np.mean(losses_velocity)
            pos_batch_loss = np.mean(losses_position)
            dis_batch_loss = np.mean(losses_distance)

            print(
                'batch: {:<8} total loss: {:<8.6f} | img loss: {:<8.6f} | iou loss: {:<8.6f} | vel loss: {:<8.6f} | pos loss: {:<8.6f} | edge loss: {:<8.6f} time(s): {:<10.2f} '
                .format(cur_batch_it, batch_loss, img_batch_loss, iou_batch_loss, vel_batch_loss, pos_batch_loss,
                        dis_batch_loss, elapsed_since_last_log)
                )
            summaries_dict = {prefix + '_total_loss': batch_loss,
                              prefix + '_img_loss': img_batch_loss,
                              prefix + '_iou_loss': iou_batch_loss,
                              prefix + '_velocity_loss': vel_batch_loss,
                              prefix + '_position_loss': pos_batch_loss,
                              prefix + '_distance_loss': dis_batch_loss
                              }
            self.logger.summarize(cur_batch_it, summaries_dict=summaries_dict, summarizer="train")

        return cur_batch_it

    def train_multiple_batches(self, prefix, n_batches_trained, n_batches_trained_since_last_save):
        features = self.sess.run(self.next_element_train)

        features = convert_dict_to_list_subdicts(features, self.config.train_batch_size)

        input_graphs_batches, target_graphs_batches = create_graphs(config=self.config,
                                                                    batch_data=features,
                                                                    initial_pos_vel_known=self.config.initial_pos_vel_known,
                                                                    batch_processing=True
                                                                    )

        if len(input_graphs_batches[-1]) != self.config.train_batch_size:
            input_graphs_batches = input_graphs_batches[:-1]
            target_graphs_batches = target_graphs_batches[:-1]

        for input_batch, target_batch in zip(input_graphs_batches, target_graphs_batches):
            start_time = time.time()
            last_log_time = start_time

            total_loss, _, loss_img, loss_iou, loss_velocity, loss_position, loss_distance, _ = self.do_step(input_batch, target_batch, features, train=True, batch_processing=True)

            self.sess.run(self.model.increment_cur_batch_tensor)
            cur_batch_it = self.model.cur_batch_tensor.eval(self.sess)

            the_time = time.time()
            elapsed_since_last_log = the_time - last_log_time

            print(
                'batch: {:<8} total loss: {:<8.6f} | img loss: {:<8.6f} | iou loss: {:<8.6f} | vel loss: {:<8.6f} | pos loss: {:<8.6f} | edge loss: {:<8.6f} time(s): {:<10.2f} '
                    .format(cur_batch_it, total_loss, loss_img, loss_iou, loss_velocity, loss_position,
                            loss_distance, elapsed_since_last_log)
            )
            summaries_dict = {prefix + '_total_loss': total_loss,
                              prefix + '_img_loss': loss_img,
                              prefix + '_iou_loss': loss_iou,
                              prefix + '_velocity_loss': loss_velocity,
                              prefix + '_position_loss': loss_position,
                              prefix + '_distance_loss': loss_distance
                              }
            self.logger.summarize(cur_batch_it, summaries_dict=summaries_dict, summarizer="train")

        return cur_batch_it, n_batches_trained + len(input_graphs_batches), n_batches_trained_since_last_save + len(input_graphs_batches)

    def test_batch(self, prefix, initial_pos_vel_known, export_images=False, process_all_nn_outputs=False, sub_dir_name=None,
                   export_latent_data=False, output_results=True):

        losses_total = []
        losses_img = []
        losses_iou = []
        losses_velocity = []
        losses_position = []
        losses_distance = []
        outputs_total = []
        targets_total = []
        exp_id_total = []
        summaries_dict = {}
        summaries_dict_images = {}

        features = self.sess.run(self.next_element_test)

        features = convert_dict_to_list_subdicts(features, self.config.test_batch_size)

        start_time = time.time()
        last_log_time = start_time

        for i in range(self.config.test_batch_size):
            input_graphs_all_exp, target_graphs_all_exp = create_graphs(config=self.config,
                                                                        batch_data=features[i],
                                                                        initial_pos_vel_known=self.config.initial_pos_vel_known,
                                                                        batch_processing=False
                                                                        )
            output_i = []
            target_i = []
            exp_id_i = []


            for j in range(features[i]["unpadded_experiment_length"] - 1):
                total_loss, output, loss_img, loss_iou, loss_velocity, loss_position, loss_distance, target = self.do_step(input_graphs_all_exp[j],
                                                                                                       target_graphs_all_exp[j],
                                                                                                       features[i],
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
                target_i.append(target)
                exp_id_i.append(features[i]['experiment_id'])


            outputs_total.append((output_i, i))
            targets_total.append((target_i, i))
            exp_id_total.append(exp_id_i)

        the_time = time.time()
        elapsed_since_last_log = the_time - last_log_time
        cur_batch_it = self.model.cur_batch_tensor.eval(self.sess)

        if not process_all_nn_outputs:
            """ due to brevity, just use last output """
            outputs_total = [outputs_total[-1]]

        if losses_total:
            batch_loss = np.mean(losses_total)
            img_batch_loss = np.mean(losses_img)
            iou_batch_loss = np.mean(losses_iou)
            vel_batch_loss = np.mean(losses_velocity)
            pos_batch_loss = np.mean(losses_position)
            dis_batch_loss = np.mean(losses_distance)

            print('total test batch loss: {:<8.6f} | img loss: {:<8.6f} | iou loss: {:<8.6f} | vel loss: {:<8.6f} | pos loss {:<8.6f} | edge loss {:<8.6f} time(s): {:<10.2f}'.format(
                batch_loss, img_batch_loss, iou_batch_loss, vel_batch_loss, pos_batch_loss, dis_batch_loss, elapsed_since_last_log))

            summaries_dict = {prefix + '_total_loss': batch_loss,
                              prefix + '_img_loss': img_batch_loss,
                              prefix + '_iou_loss': iou_batch_loss,
                              prefix + '_velocity_loss': vel_batch_loss,
                              prefix + '_position_loss': pos_batch_loss,
                              prefix + '_edge_loss': dis_batch_loss
                              }

        else:
            batch_loss, img_batch_loss, vel_batch_loss, pos_batch_loss, dis_batch_loss, iou_batch_loss = 0, 0, 0, 0, 0, 0

        if outputs_total and output_results:
            if self.config.parallel_batch_processing:
                with parallel_backend('threading', n_jobs=-2):
                    results = Parallel()(delayed(generate_results)(output,
                                                                      self.config,
                                                                      prefix,
                                                                      features,
                                                                      cur_batch_it,
                                                                      export_images,
                                                                      export_latent_data,
                                                                      sub_dir_name,
                                                                      True,
                                                                      ['seg']) for output in outputs_total)

            else:
                for output in outputs_total:
                    summaries_dict_images, summaries_pos_dict_images, _ = generate_results(output=output,
                                                                         config=self.config,
                                                                         prefix=prefix,
                                                                         features=features,
                                                                         cur_batch_it=cur_batch_it,
                                                                         export_images=export_images,
                                                                         export_latent_data=export_latent_data,
                                                                         dir_name=sub_dir_name,
                                                                         reduce_dict=True,
                                                                         output_selection=['seg', 'rgb', 'depth']
                                                                        )

            if summaries_dict_images:
                if self.config.parallel_batch_processing:
                    """ parallel mode returns list, just use first element as a summary for the logger """
                    summaries_dict_images = results[0]
                    summaries_pos_dict_images = results[1]

                    summaries_dict_images = summaries_dict_images[0]
                    if summaries_pos_dict_images is not None: 
                        summaries_pos_dict_images = summaries_pos_dict_images[0]

                if summaries_pos_dict_images is not None:
                    summaries_dict = {**summaries_dict, **summaries_dict_images, **summaries_pos_dict_images}
                else:
                    summaries_dict = {**summaries_dict, **summaries_dict_images}
                cur_batch_it = self.model.cur_batch_tensor.eval(self.sess)
                self.logger.summarize(cur_batch_it, summaries_dict=summaries_dict, summarizer="test")

        return batch_loss, img_batch_loss, vel_batch_loss, pos_batch_loss, dis_batch_loss, iou_batch_loss, cur_batch_it, outputs_total, targets_total, exp_id_total

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
                            total_loss, output, loss_img, loss_iou, loss_velocity, loss_position, loss_distance, _ = self.do_step(input_graphs_all_exp[j],
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

    def test_statistics(self, prefix, initial_pos_vel_known, export_images=False, sub_dir_name="test_statistics", export_latent_data=True):

        if not self.config.n_epochs == 1:
            print("test statistics mode --> n_epochs will be set to 1")
            self.config.n_epochs = 1

        reduce_dict = False
        overlay_images = True
        pos_means = []
        pos_std = []
        vel_means = []
        vel_std = []
        psnr = []

        create_dir(os.path.join("../experiments", prefix), sub_dir_name)
        while True:
            try:
                outputs_total = []
                features = self.sess.run(self.next_element_test)

                features = convert_dict_to_list_subdicts(features, self.config.test_batch_size)

                input_graphs_all_exp, target_graphs_all_exp, input_ctrl_graphs_all_exp = create_graphs(config=self.config,
                                                                                                       batch_data=features,
                                                                                                       batch_size=self.config.test_batch_size,
                                                                                                       initial_pos_vel_known=initial_pos_vel_known
                                                                                                       )

                for i in range(self.config.test_batch_size):
                    _, output, _, _, _, _, _, _ = self.do_step(input_graphs_all_exp[i],
                                                                    target_graphs_all_exp[i],
                                                                    input_ctrl_graphs_all_exp[i],
                                                                    features[i],
                                                                    train=False
                                                                    )
                    if output:
                        outputs_total.append((output, i))

                cur_batch_it = self.model.cur_batch_tensor.eval(self.sess)

                if outputs_total:
                    if self.config.parallel_batch_processing:
                        with parallel_backend('loky', n_jobs=-2):
                            results = Parallel()(delayed(generate_results)(output,
                                                                self.config,
                                                                prefix,
                                                                features,
                                                                cur_batch_it,
                                                                export_images,
                                                                export_latent_data,
                                                                sub_dir_name,
                                                                reduce_dict,
                                                                overlay_images
                                                                ) for output in outputs_total)

                        for result in results:
                            images_dict = result[0]
                            df = result[2]

                            psnr.append(compute_psnr(images_dict))
                            #print(images_dict, df)

                    else:
                        for output in outputs_total:
                            summaries_dict_images, _, df = generate_results(output=output,
                                                                            config=self.config,
                                                                            prefix=prefix,
                                                                            features=features,
                                                                            cur_batch_it=cur_batch_it,
                                                                            export_images=export_images,
                                                                            export_latent_data=export_latent_data,
                                                                            dir_name=sub_dir_name,
                                                                            reduce_dict=True,
                                                                            overlay_images=overlay_images
                                                                            )
                summaries_dict_images = None
                df = None

            except tf.errors.OutOfRangeError:
                break
            else:
                print("continue")
                continue


    def _do_step_parallel(self, input_graph, target_graphs, input_ctrl_graph, features, outputs, losses, losses_img, losses_iou, losses_velocity, losses_position, losses_distance, yield_outputs=False, train=False, compute_losses=False):
        total_loss, output, loss_img, loss_iou, loss_velocity, loss_position, loss_distance = self.do_step(input_graph, target_graphs, input_ctrl_graph, features, train=train)
        if total_loss is not None and compute_losses:
            losses.append(total_loss)
            losses_img.append(loss_img)
            losses_iou.append(loss_iou)
            losses_velocity.append(loss_velocity)
            losses_position.append(loss_position)
            losses_distance.append(loss_distance)
        if yield_outputs:
            outputs.append(output)
        return losses, outputs, losses_img, losses_iou, losses_velocity, losses_position, losses_distance


