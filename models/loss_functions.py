import tensorflow as tf
from utils.utils import get_correct_image_shape


def create_loss_ops(config, target_op, output_ops):
    """ ground truth nodes are given by tensor target_op of shape (n_nodes*experience_length, node_output_size) but output_ops
    is a list of graph tuples with shape (n_nodes, node_output_size) --> split at the first dimension in order to compute node-wise MSE error
    --> same applies for edges
    we further assume object/node features are given in the following order (pixels/visual information, velocity(3dim), pos (3dim))

    """
    #mult = tf.constant([len(output_ops)])
    #n_nodes = [tf.shape(output_ops[0].nodes)[0]]
    #n_edges = [tf.shape(output_ops[0].edges)[0]]

    #target_node_splits = tf.split(target_op.nodes, num_or_size_splits=tf.tile(n_nodes, mult), axis=0)
    #target_edge_splits = tf.split(target_op.edges, num_or_size_splits=tf.tile(n_edges, mult), axis=0)

    """ if object seg data is only used for init, the ground truth features in the rest of the sequence are static except position 
    --> in this case compute loss only over the position since image prediction is infeasible """

    total_loss_ops = []
    loss_ops_img = []
    loss_ops_position = []
    loss_ops_velocity = []
    loss_ops_distance = []
    loss_ops_img_iou = []

    img_scale = 1
    non_visual_scale = 1

    if config.loss_type == 'cross_entropy_seg_only':
        for i, output_op in enumerate(output_ops):
            ''' checks whether the padding_flag is 1 or 0 --> if 1, return inf loss (later removed)'''
            condition = tf.equal(target_op.globals[i, 0], tf.constant(1.0))

            """ VISUAL LOSS """
            predicted_node_reshaped = _transform_into_images(config, output_op.nodes, img_type="seg")  # returns shape (?, 120,160,1)
            target_node_reshaped = _transform_into_images(config, target_op.nodes, img_type="all")  # returns shape (?, 120,160,7)
            segmentation_data_predicted = predicted_node_reshaped[:, :, :, 0]  # --> transform into (?,120,160)
            segmentation_data_gt = target_node_reshaped[:, :, :, 3]  # --> transform into (?,120,160)

            logits = tf.reshape(segmentation_data_predicted, [-1,
                                                              segmentation_data_predicted.get_shape()[1] *
                                                              segmentation_data_predicted.get_shape()[2]])
            labels = tf.reshape(segmentation_data_gt, [-1,
                                                       segmentation_data_gt.get_shape()[1] *
                                                       segmentation_data_gt.get_shape()[2]])


            loss_visual_mse_nodes = tf.cond(condition, lambda: float("inf"),
                                                lambda: img_scale * 1.0 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                                        labels=labels,
                                                        logits=logits))
                                            )

            tf.losses.add_loss(loss_visual_mse_nodes)

            loss_visual_iou_seg = 0.0  # no iou loss computed

            """ NONVISUAL LOSS (50% weight) """
            loss_nonvisual_mse_edges = tf.cond(condition, lambda: float("inf"),
                                               lambda: non_visual_scale * tf.losses.mean_squared_error(
                                                   labels=target_op.edges,
                                                   predictions=output_op.edges,
                                                   weights=0.1)
                                               )

            loss_nonvisual_mse_edges = 0.0

            loss_nonvisual_mse_nodes_pos = tf.cond(condition, lambda: float("inf"),
                                                   lambda: non_visual_scale * tf.losses.mean_squared_error(
                                                       labels=target_op.nodes[:, -3:],
                                                       predictions=output_op.nodes[:, -3:],
                                                       weights=0.3)
                                                   )
            loss_nonvisual_mse_nodes_vel = tf.cond(condition, lambda: float("inf"),
                                                   lambda: tf.losses.mean_squared_error(
                                                       labels=target_op.nodes[:, -6:-3:],
                                                       predictions=output_op.nodes[:, -6:-3:],
                                                       weights=0.1)
                                                   )

            loss_ops_img.append(loss_visual_mse_nodes)
            loss_ops_img_iou.append(loss_visual_iou_seg)
            loss_ops_velocity.append(loss_nonvisual_mse_nodes_vel)
            loss_ops_position.append(loss_nonvisual_mse_nodes_pos)
            loss_ops_distance.append(loss_nonvisual_mse_edges)

            print("---- image loss only ----")
            total_loss_ops.append(loss_visual_mse_nodes)
            #total_loss_ops.append(loss_visual_mse_nodes + loss_nonvisual_mse_edges + loss_nonvisual_mse_nodes_vel + loss_nonvisual_mse_nodes_pos)


    else:
        raise ValueError("loss type must be in [\"mse\", \"mse_gdl\", \"mse_iou\", \"mse_seg_only\", \"cross_entropy_seg_only\" but is {}".format(config.loss_type))

    l2_loss = tf.losses.get_regularization_loss()
    total_loss_ops += l2_loss

    return total_loss_ops, loss_ops_img, loss_ops_img_iou, loss_ops_velocity, loss_ops_position, loss_ops_distance

def _transform_into_images(config, data, img_type="all"):
    """ reshapes data (shape: (batch_size, feature_length)) into the required image shape with an
    additional batch_dimension, e.g. (1,120,160,7) """
    data_shape = get_correct_image_shape(config, get_type=img_type)
    data = data[:, :-6]
    data = tf.reshape(data, [-1, *data_shape])
    return data
