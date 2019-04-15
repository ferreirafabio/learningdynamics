import numpy as np
import tensorflow as tf
import time
import os
from base.base_train import BaseTrain
from utils.conversions import convert_dict_to_list_subdicts
from utils.tf_summaries import generate_results
from utils.io import create_dir
from utils.math_ops import sigmoid
from models.singulation_graph import create_graphs, create_feed_dict
from joblib import parallel_backend, Parallel, delayed
from eval.compute_test_run_statistics import compute_psnr

class SingulationTrainer(BaseTrain):
    def __init__(self, sess, model, train_data, valid_data, config, logger, only_test):
        super(SingulationTrainer, self).__init__(sess, model, train_data, valid_data, config, logger, only_test=only_test)

    def train_epoch(self):
        prefix = self.config.exp_name
        print("using {} rollout steps (n_rollouts={})".format(self.config.n_rollouts-1, self.config.n_rollouts))

        while True:
            try:
                _, _, _, _, _, cur_batch_it = self.train_batch(prefix)
                if cur_batch_it % self.config.model_save_step_interval == 1:
                    self.model.save(self.sess)
                if cur_batch_it % self.config.test_interval == 1:
                    print("Executing test batch")
                    self.test_batch(prefix, export_images=self.config.export_test_images,
                                    initial_pos_vel_known=self.config.initial_pos_vel_known,
                                    sub_dir_name="tests_during_training")

            except tf.errors.OutOfRangeError:
                break

    def do_step(self, input_graph, target_graphs, input_ctrl_graphs, feature, train=True, convert_seg_to_unit_step=True):

        if train:
            feed_dict = create_feed_dict(self.model.input_ph, self.model.target_ph, self.model.input_ctrl_ph,
                                         input_graph, target_graphs, input_ctrl_graphs)

            data = self.sess.run({"step": self.model.step_op,
                                  "target": self.model.target_ph,
                                  "loss_total": self.model.loss_op_train_total,
                                  "outputs": self.model.output_ops_train,
                                  "loss_img": self.model.loss_ops_train_img,
                                  "loss_iou": self.model.loss_ops_train_iou,
                                  "loss_velocity": self.model.loss_ops_train_velocity,
                                  "loss_position": self.model.loss_ops_train_position,
                                  "loss_distance": self.model.loss_ops_train_distance,
                                  "targ_train": self.model.targ_train
                                  }, feed_dict=feed_dict)

        else:
            feed_dict = create_feed_dict(self.model.input_ph_test, self.model.target_ph_test, self.model.input_ctrl_ph_test,
                                         input_graph, target_graphs, input_ctrl_graphs)

            data = self.sess.run({"target": self.model.target_ph_test,
                                  "loss_total": self.model.loss_op_test_total,
                                  "outputs": self.model.output_ops_test,
                                  "loss_img": self.model.loss_ops_test_img,
                                  "loss_iou": self.model.loss_ops_test_iou,
                                  "loss_velocity": self.model.loss_ops_test_velocity,
                                  "loss_position": self.model.loss_ops_test_position,
                                  "loss_distance": self.model.loss_ops_test_distance,
                                  "targ_test": self.model.targ_test
                                  }, feed_dict=feed_dict)

            if convert_seg_to_unit_step:
                """ currently we are interested in values in {0,1} since we train binary cross entropy (segmentation images). 
                tf.nn.sigmoid_cross_entropy_with_logits runs the logits through sigmoid() which is why for outputs we also 
                need to run a sigmoid(). """
                for output in data['outputs']:
                    seg_data = output.nodes[:, :-6]
                    seg_data = sigmoid(seg_data)
                    seg_data[seg_data >= 0.5] = 1.0
                    seg_data[seg_data < 0.5] = 0.0
                    output.nodes[:, :-6] = seg_data

        return data['loss_total'], data['outputs'], data['loss_img'], data['loss_iou'], data['loss_velocity'], data['loss_position'], data['loss_distance']


    def test_rollouts(self):
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
                                sub_dir_name="test_{}_rollouts_{}_iterations_trained".format(self.config.n_rollouts, cur_batch_it))
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

        if self.config.do_multi_step_prediction_at_test:
            sub_dir_name = "test_5_objects_multi_step_{}_iterations_trained".format(cur_batch_it)
        else:
            sub_dir_name = "test_5_objects_single_step_{}_iterations_trained".format(cur_batch_it)

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

        next_element = self.train_data.get_next_batch()
        features = self.sess.run(next_element)

        features = convert_dict_to_list_subdicts(features, self.config.train_batch_size)
        input_graphs_all_exp, target_graphs_all_exp, input_ctrl_graphs_all_exp = create_graphs(config=self.config,
                                                                                    batch_data=features,
                                                                                    batch_size=self.config.train_batch_size,
                                                                                    initial_pos_vel_known=self.config.initial_pos_vel_known
                                                                                    )

        start_time = time.time()
        last_log_time = start_time

        for i in range(self.config.train_batch_size):
            total_loss, _, loss_img, loss_iou, loss_velocity, loss_position, loss_distance = self.do_step(input_graphs_all_exp[i],
                                                                                                target_graphs_all_exp[i],
                                                                                                input_ctrl_graphs_all_exp[i],
                                                                                                features[i])
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

            print('batch: {:<8} total loss: {:<8.6f} | img loss: {:<8.6f} | iou loss: {:<8.6f} | vel loss: {:<8.6f} | pos loss: {:<8.6f} | distance loss: {:<8.6f} time(s): {:<10.2f} '
                .format(cur_batch_it, batch_loss, img_batch_loss, iou_batch_loss, vel_batch_loss, pos_batch_loss, dis_batch_loss, elapsed_since_last_log)
            )
            summaries_dict = {prefix + '_total_loss': batch_loss,
                              prefix + '_img_loss': img_batch_loss,
                              prefix + '_iou_loss': iou_batch_loss,
                              prefix + '_velocity_loss': vel_batch_loss,
                              prefix + '_position_loss': pos_batch_loss,
                              prefix + '_distance_loss': dis_batch_loss
                              }
            self.logger.summarize(cur_batch_it, summaries_dict=summaries_dict, summarizer="train")
        else:
            batch_loss, img_batch_loss, vel_batch_loss, pos_batch_loss, dis_batch_loss, iou_batch_loss = 0, 0, 0, 0, 0, 0

        return batch_loss, vel_batch_loss, pos_batch_loss, dis_batch_loss, iou_batch_loss, cur_batch_it

    def test_batch(self, prefix, initial_pos_vel_known, export_images=False, process_all_nn_outputs=False, sub_dir_name=None,
                   export_latent_data=True):

        losses_total = []
        losses_img = []
        losses_iou = []
        losses_velocity = []
        losses_position = []
        losses_distance = []
        outputs_total = []
        summaries_dict = {}
        summaries_dict_images = {}

        next_element = self.test_data.get_next_batch()
        features = self.sess.run(next_element)

        features = convert_dict_to_list_subdicts(features, self.config.test_batch_size)

        input_graphs_all_exp, target_graphs_all_exp, input_ctrl_graphs_all_exp = create_graphs(config=self.config,
                                                                                    batch_data=features,
                                                                                    batch_size=self.config.test_batch_size,
                                                                                    initial_pos_vel_known=initial_pos_vel_known
                                                                                    )

        start_time = time.time()
        last_log_time = start_time

        for i in range(self.config.test_batch_size):
            total_loss, output, loss_img, loss_iou, loss_velocity, loss_position, loss_distance = self.do_step(input_graphs_all_exp[i],
                                                                                                      target_graphs_all_exp[i],
                                                                                                      input_ctrl_graphs_all_exp[i],
                                                                                                      features[i],
                                                                                                      train=False
                                                                                                      )
            if total_loss is not None:
                losses_total.append(total_loss)
                losses_img.append(loss_img)
                losses_iou.append(loss_iou)
                losses_velocity.append(loss_velocity)
                losses_position.append(loss_position)
                losses_distance.append(loss_distance)

            if output:
                outputs_total.append((output, i))

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

            print('total test batch loss: {:<8.6f} | img loss: {:<8.6f} | iou loss: {:<8.6f} | vel loss: {:<8.6f} | pos loss {:<8.6f} | distance loss {:<8.6f} time(s): {:<10.2f}'.format(
                batch_loss, img_batch_loss, iou_batch_loss, vel_batch_loss, pos_batch_loss, dis_batch_loss, elapsed_since_last_log))

            summaries_dict = {prefix + '_total_loss': batch_loss,
                              prefix + '_img_loss': img_batch_loss,
                              prefix + '_iou_loss': iou_batch_loss,
                              prefix + '_velocity_loss': vel_batch_loss,
                              prefix + '_position_loss': pos_batch_loss,
                              prefix + '_distance_loss': dis_batch_loss
                              }

        else:
            batch_loss, img_batch_loss, vel_batch_loss, pos_batch_loss, dis_batch_loss, iou_batch_loss = 0, 0, 0, 0, 0, 0

        if outputs_total:
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

        return batch_loss, vel_batch_loss, pos_batch_loss, dis_batch_loss, iou_batch_loss, cur_batch_it

    def test_specific_exp_ids(self):
        assert self.config.n_epochs == 1, "set n_epochs to 1 for test mode"
        prefix = self.config.exp_name
        print("Running tests with initial_pos_vel_known={}".format(self.config.initial_pos_vel_known))
        cur_batch_it = self.model.cur_batch_tensor.eval(self.sess)

        exp_ids_to_export = [14573, 15671, 11699, 11529, 14293, 10765, 1143, 19859, 8388, 14616, 16854, 17272, 1549,
                           8961, 14756, 11167, 18828, 10689, 17192, 10512, 10667]

        export_images = self.config.export_test_images,
        initial_pos_vel_known = self.config.initial_pos_vel_known
        export_latent_data = True
        process_all_nn_outputs = True
        sub_dir_name = "test_{}_specific_exp_ids_{}_iterations_trained".format(self.config.n_rollouts, cur_batch_it)

        while True:
            try:
                losses_total = []
                losses_img = []
                losses_iou = []
                losses_velocity = []
                losses_position = []
                losses_distance = []
                outputs_total = []

                next_element = self.test_data.get_next_batch()
                features = self.sess.run(next_element)

                features = convert_dict_to_list_subdicts(features, self.config.test_batch_size)

                if exp_ids_to_export:
                    features_to_export = []
                    for dct in features:
                        if dct["experiment_id"] in exp_ids_to_export:
                            features_to_export.append(dct)
                            print("added", dct["experiment_id"])

                    features = features_to_export

                if exp_ids_to_export and not features_to_export:
                    return

                # if add_noise_to_gripper:
                #     print("adding noise to the gripperpos")
                #     for dct in features:
                #         #dct['gripperpos'] = dct['gripperpos'] + np.random.normal(0, 1.0, (10, 3))
                #         dct['objpos'] = dct['objpos'] + np.random.normal(0, 1.0, (10, 3, 3))

                input_graphs_all_exp, target_graphs_all_exp, input_ctrl_graphs_all_exp = create_graphs(config=self.config,
                                                                                                       batch_data=features,
                                                                                                       batch_size=self.config.test_batch_size,
                                                                                                       initial_pos_vel_known=initial_pos_vel_known)

                start_time = time.time()
                last_log_time = start_time

                for i in range(self.config.test_batch_size):
                    total_loss, outputs, loss_img, loss_iou, loss_velocity, loss_position, loss_distance = self.do_step(input_graphs_all_exp[i],
                                                                                                              target_graphs_all_exp[i],
                                                                                                              input_ctrl_graphs_all_exp[
                                                                                                                  i], features[i],
                                                                                                              train=False)
                    if total_loss is not None:
                        losses_total.append(total_loss)
                        losses_img.append(loss_img)
                        losses_iou.append(loss_iou)
                        losses_velocity.append(loss_velocity)
                        losses_position.append(loss_position)
                        losses_distance.append(loss_distance)

                    if outputs:
                        outputs_total.append((outputs, i))

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
                next_element = self.test_data.get_next_batch()
                features = self.sess.run(next_element)

                features = convert_dict_to_list_subdicts(features, self.config.test_batch_size)

                input_graphs_all_exp, target_graphs_all_exp, input_ctrl_graphs_all_exp = create_graphs(config=self.config,
                                                                                                       batch_data=features,
                                                                                                       batch_size=self.config.test_batch_size,
                                                                                                       initial_pos_vel_known=initial_pos_vel_known
                                                                                                       )

                for i in range(self.config.test_batch_size):
                    _, output, _, _, _, _, _ = self.do_step(input_graphs_all_exp[i],
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


