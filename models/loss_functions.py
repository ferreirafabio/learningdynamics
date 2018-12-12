import tensorflow as tf
from utils.utils import get_correct_image_shape


def create_loss_ops(config, target_op, output_ops):
    """ ground truth nodes are given by tensor target_op of shape (n_nodes*experience_length, node_output_size) but output_ops
    is a list of graph tuples with shape (n_nodes, exp_len) --> split at the first dimension in order to compute node-wise MSE error
    --> same applies for edges """
    mult = tf.constant([len(output_ops)])
    n_nodes = [tf.shape(output_ops[0].nodes)[0]]
    n_edges = [tf.shape(output_ops[0].edges)[0]]

    target_node_splits = tf.split(target_op.nodes, num_or_size_splits=tf.tile(n_nodes, mult), axis=0)
    target_edge_splits = tf.split(target_op.edges, num_or_size_splits=tf.tile(n_edges, mult), axis=0)

    """ if object seg data is only used for init, the ground truth features in the rest of the sequence are static except position 
    --> in this case compute loss only over the position since image prediction is infeasible """
    pos_vel_loss_ops = [tf.losses.mean_squared_error(output_op.nodes[:, -6:], target_node_splits[i][:, -6:]) for i, output_op in
                        enumerate(output_ops)]

    if config.use_object_seg_data_only_for_init:
        # compute loss of the nodes only over velocity and position and not over ground truth static images
        loss_ops = [
            tf.losses.mean_squared_error(output_op.nodes[:, -6:], target_node_splits[i][:, -6:]) +
            tf.losses.mean_squared_error(output_op.edges, target_edge_splits[i])
            for i, output_op in enumerate(output_ops)]
    else:
        if config.loss_type == 'mse':
            loss_ops = [tf.losses.mean_squared_error(output_op.nodes, target_node_splits[i]) +
                        tf.losses.mean_squared_error(output_op.edges, target_edge_splits[i])
                        for i, output_op in enumerate(output_ops)
            ]
        elif config.loss_type == 'mse_gdl':
            loss_ops = []
            for i, output_op in enumerate(output_ops):
                """ MSE loss (frames and non-visual data) """
                loss_mse_edges = tf.losses.mean_squared_error(output_op.edges, target_edge_splits[i])
                loss_mse_nodes = 0.5 * tf.losses.mean_squared_error(output_op.nodes, target_node_splits[i])

                """ Gradient difference loss (frames only) """
                predicted_node_reshaped = _transform_into_images(config, output_op.nodes)
                target_node_reshaped = _transform_into_images(config, target_node_splits[i])
                loss_gdl_nodes = 0.5 * gradient_difference_loss(predicted_node_reshaped, target_node_reshaped)

                pos_vel_loss_ops = tf.losses.mean_squared_error(output_op.nodes[:, -6:], target_node_splits[i][:, -6:])

                loss_ops.append(loss_mse_edges + loss_mse_nodes + loss_gdl_nodes + pos_vel_loss_ops)

    return loss_ops, pos_vel_loss_ops


def gradient_difference_loss(true, pred, alpha=2.0):
    """
    computes gradient difference loss of two images
    :param ground truth image: Tensor of shape (batch_size, frame_height, frame_width, num_channels)
    :param predicted image: Tensor of shape (batch_size, frame_height, frame_width, num_channels)
    :param alpha parameter of the used l-norm
    """
    if true.get_shape()[3] == 7:
        true_pred_diff_vert_rgb = tf.pow(tf.abs(difference_gradient(true[..., :3], vertical=True) - difference_gradient(pred[..., :3], vertical=True)), alpha)
        true_pred_diff_hor_rgb = tf.pow(tf.abs(difference_gradient(true[..., :3], vertical=False) - difference_gradient(pred[..., :3], vertical=False)), alpha)
        true_pred_diff_vert_seg = tf.pow(tf.abs(difference_gradient(true[..., 4, None], vertical=True) - difference_gradient(pred[..., 4, None], vertical=True)), alpha)
        true_pred_diff_hor_seg = tf.pow(tf.abs(difference_gradient(true[..., 4, None], vertical=False) - difference_gradient(pred[..., 4, None], vertical=False)), alpha)
        true_pred_diff_vert_depth = tf.pow(tf.abs(difference_gradient(true[..., -3:], vertical=True) - difference_gradient(pred[..., -3:], vertical=True)), alpha)
        true_pred_diff_hor_depth = tf.pow(tf.abs(difference_gradient(true[..., -3:], vertical=False) - difference_gradient(pred[..., -3:], vertical=False)), alpha)

        return (tf.reduce_mean(true_pred_diff_vert_rgb) + tf.reduce_mean(true_pred_diff_hor_rgb)) / tf.to_float(2) + \
               (tf.reduce_mean(true_pred_diff_vert_seg) + tf.reduce_mean(true_pred_diff_hor_seg)) / tf.to_float(2) + \
               (tf.reduce_mean(true_pred_diff_vert_depth) + tf.reduce_mean(true_pred_diff_hor_depth)) / tf.to_float(2)

    else:
        #tf.assert_equal(tf.shape(true), tf.shape(pred))
        """ vertical """
        true_pred_diff_vert = tf.pow(tf.abs(difference_gradient(true, vertical=True) - difference_gradient(pred, vertical=True)), alpha)
        """ horizontal """
        true_pred_diff_hor = tf.pow(tf.abs(difference_gradient(true, vertical=False) - difference_gradient(pred, vertical=False)), alpha)
        """ normalization over all dimensions """
        return (tf.reduce_mean(true_pred_diff_vert) + tf.reduce_mean(true_pred_diff_hor)) / tf.to_float(2)



def difference_gradient(image, vertical=True):
  """
  :param image: Tensor of shape (batch_size, frame_height, frame_width, num_channels)
  :param vertical: boolean that indicates whether vertical or horizontal pixel gradient shall be computed
  :return: difference_gradient -> Tenor of shape (:, frame_height-1, frame_width, :) if vertical and (:, frame_height, frame_width-1, :) else
  """
  s = tf.shape(image)
  if vertical:
    return tf.abs(image[:, 0:s[1] - 1, :, :] - image[:, 1:s[1], :, :])
  else:
    return tf.abs(image[:, :, 0:s[2]-1, :] - image[:, :, 1:s[2], :])


def _transform_into_images(config, data):
    """ reshapes data (shape: (batch_size, feature_length)) into the required image shape with an
    additional batch_dimension, e.g. (1,120,160,7) """
    data_shape = get_correct_image_shape(config, get_type="all")
    data = data[:, :-6]
    data = tf.reshape(data, [-1, *data_shape])
    return data
