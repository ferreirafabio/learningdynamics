import tensorflow as tf
from utils.utils import get_correct_image_shape


def create_loss_ops(config, target_op, output_ops):
    """ ground truth nodes are given by tensor target_op of shape (n_nodes*experience_length, node_output_size) but output_ops
    is a list of graph tuples with shape (n_nodes, exp_len) --> split at the first dimension in order to compute node-wise MSE error
    --> same applies for edges
    we further assume object/node features are given in the following order (pixels/visual information, velocity(3dim), pos (3dim))

    """
    mult = tf.constant([len(output_ops)])
    n_nodes = [tf.shape(output_ops[0].nodes)[0]]
    n_edges = [tf.shape(output_ops[0].edges)[0]]

    target_node_splits = tf.split(target_op.nodes, num_or_size_splits=tf.tile(n_nodes, mult), axis=0)
    target_edge_splits = tf.split(target_op.edges, num_or_size_splits=tf.tile(n_edges, mult), axis=0)

    """ if object seg data is only used for init, the ground truth features in the rest of the sequence are static except position 
    --> in this case compute loss only over the position since image prediction is infeasible """

    total_loss_ops = []
    loss_ops_img = []
    loss_ops_position = []
    loss_ops_velocity = []
    loss_ops_distance = []
    loss_ops_img_iou = []

    if config.loss_type == 'mse':
        for i, output_op in enumerate(output_ops):
            """ VISUAL LOSS """
            loss_visual_mse_nodes = tf.losses.mean_squared_error(output_op.nodes[:, :-6], target_node_splits[i][:, :-6], weights=0.5)

            """ NONVISUAL LOSS (50% weight) """
            loss_nonvisual_mse_edges = tf.losses.mean_squared_error(output_op.edges, target_edge_splits[i], weights=(1/6))
            loss_nonvisual_mse_nodes_pos = tf.losses.mean_squared_error(output_op.nodes[:, -3:], target_node_splits[i][:, -3:], weights=(1/6))
            loss_nonvisual_mse_nodes_vel = tf.losses.mean_squared_error(output_op.nodes[:, -6:-3:], target_node_splits[i][:, -6:-3:], weights=(1/6))

            loss_ops_img.append(loss_visual_mse_nodes)
            loss_ops_velocity.append(loss_nonvisual_mse_nodes_vel)
            loss_ops_position.append(loss_nonvisual_mse_nodes_pos)
            loss_ops_distance.append(loss_nonvisual_mse_edges)

            total_loss_ops.append(loss_visual_mse_nodes + loss_nonvisual_mse_edges + loss_nonvisual_mse_nodes_vel + loss_nonvisual_mse_nodes_pos)

    elif config.loss_type == 'mse_iou':
        for i, output_op in enumerate(output_ops):

            """ VISUAL LOSS """
            # if depth = True, predicted_node_reshaped and target_node_reshape have shape (120,160,7)
            predicted_node_reshaped = _transform_into_images(config, output_op.nodes)[0]
            target_node_reshaped = _transform_into_images(config, target_node_splits[i])[0]


            segmentation_data_predicted = predicted_node_reshaped[:, :, 3]
            segmentation_data_gt = target_node_reshaped[:, :, 3]

            loss_visual_iou_seg = (1/3) * _intersection_over_union(segmentation_data_gt, segmentation_data_predicted)

            if config.depth_data_provided:
                # todo: check if stack is correct, print shapes
                image_data_predicted = tf.stack([predicted_node_reshaped[:, :, :3], predicted_node_reshaped[:, :, -3:]])
                image_data_gt = tf.stack([target_node_reshaped[:, :, :3], target_node_reshaped[:, :, -3:]])
            else:
                image_data_predicted = predicted_node_reshaped[:, :, :3]
                image_data_gt = target_node_reshaped[:, :, :3]

            loss_visual_mse_nodes = tf.losses.mean_squared_error(image_data_predicted, image_data_gt, weights=(2/3))

            """ NONVISUAL LOSS (50% weight) """
            loss_nonvisual_mse_edges = tf.losses.mean_squared_error(output_op.edges, target_edge_splits[i], weights=(1/6))
            loss_nonvisual_mse_nodes_pos = tf.losses.mean_squared_error(output_op.nodes[:, -3:], target_node_splits[i][:, -3:], weights=(1/6))
            loss_nonvisual_mse_nodes_vel = tf.losses.mean_squared_error(output_op.nodes[:, -6:-3:], target_node_splits[i][:, -6:-3:], weights=(1/6))

            loss_ops_img.append(loss_visual_mse_nodes)
            loss_ops_img_iou.append(loss_visual_iou_seg)
            loss_ops_velocity.append(loss_nonvisual_mse_nodes_vel)
            loss_ops_position.append(loss_nonvisual_mse_nodes_pos)
            loss_ops_distance.append(loss_nonvisual_mse_edges)

            total_loss_ops.append(loss_visual_mse_nodes + loss_visual_iou_seg + loss_nonvisual_mse_edges + loss_nonvisual_mse_nodes_vel + loss_nonvisual_mse_nodes_pos)

    elif config.loss_type == 'mse_gdl':
        for i, output_op in enumerate(output_ops):
            """ VISUAL LOSS (50% weight) """
            loss_visual_mse_nodes = tf.losses.mean_squared_error(output_op.nodes[:, :-6], target_node_splits[i][:, :-6], weights=0.25)

            # pre-processing step for gdl function call
            predicted_node_reshaped = _transform_into_images(config, output_op.nodes)
            target_node_reshaped = _transform_into_images(config, target_node_splits[i])
            loss_visual_gdl_nodes = 0.25 * gradient_difference_loss(predicted_node_reshaped, target_node_reshaped)

            """ NONVISUAL LOSS (50% weight) """
            loss_nonvisual_mse_edges = tf.losses.mean_squared_error(output_op.edges, target_edge_splits[i], weights=(1/6))
            loss_nonvisual_mse_nodes_pos = tf.losses.mean_squared_error(output_op.nodes[:, -3:], target_node_splits[i][:, -3:], weights=(1/6))
            loss_nonvisual_mse_nodes_vel = tf.losses.mean_squared_error(output_op.nodes[:, -6:-3:], target_node_splits[i][:, -6:-3:], weights=(1/6))

            loss_ops_img.append(loss_visual_gdl_nodes + loss_visual_mse_nodes)
            loss_ops_velocity.append(loss_nonvisual_mse_nodes_vel)
            loss_ops_position.append(loss_nonvisual_mse_nodes_pos)
            loss_ops_distance.append(loss_nonvisual_mse_edges)

            total_loss_ops.append(loss_visual_mse_nodes + loss_visual_gdl_nodes + loss_nonvisual_mse_edges + loss_nonvisual_mse_nodes_pos + loss_nonvisual_mse_nodes_vel)

    else:
        raise ValueError("loss type must be in [\"mse\", \"mse_gdl\" but is {}".format(config.loss_type))

    return total_loss_ops, loss_ops_img, loss_ops_img_iou, loss_ops_velocity, loss_ops_position, loss_ops_distance


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
        true_pred_diff_vert_seg = tf.pow(tf.abs(difference_gradient(true[..., 3, None], vertical=True) - difference_gradient(pred[..., 3, None], vertical=True)), alpha)
        true_pred_diff_hor_seg = tf.pow(tf.abs(difference_gradient(true[..., 3, None], vertical=False) - difference_gradient(pred[..., 3, None], vertical=False)), alpha)
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


def _intersection_over_union(image_gt, image_pred):
    """
    We assume the (x,y) coordinate system is inverted, i.e. (0,0) is on the top left, (0,1) top right, (1,1) right bottom
    :param image_gt: the ground truth segmentation image, expects a tensor of shape (120,160,1)
    :param image_gt: the predicted image, expects a tensor of shape (120,160,1)
    :return: the intersection over union. The union is computed for the bounding boxes represented by the min and max values of the segments
    """

    mask = image_pred > 0
    coordinates_pred = tf.where(mask)

    mask = image_gt > 0
    coordinates_gt = tf.where(mask)

    # axis=1 means reduce over axis 1 --> determine all minimum values over axis 0 --> yields one vector with axis 0 min values,
    # outer tf.reduce_min then extracts min single value of this vector
    x11 = tf.reduce_min(tf.reduce_min(coordinates_gt, axis=1))
    y11 = tf.reduce_min(tf.reduce_min(coordinates_gt, axis=0))
    x12 = tf.reduce_max(tf.reduce_max(coordinates_gt, axis=1))
    y12 = tf.reduce_max(tf.reduce_max(coordinates_gt, axis=0))

    x21 = tf.reduce_min(tf.reduce_min(coordinates_pred, axis=1))
    y21 = tf.reduce_min(tf.reduce_min(coordinates_pred, axis=0))
    x22 = tf.reduce_max(tf.reduce_max(coordinates_pred, axis=1))
    y22 = tf.reduce_max(tf.reduce_max(coordinates_pred, axis=0))

    # determine the (x,y) coordinates of the intersection rectangle
    xI1 = tf.maximum(x11, tf.transpose(x21))
    xI2 = tf.minimum(x12, tf.transpose(x22))

    yI1 = tf.maximum(y11, tf.transpose(y21))
    yI2 = tf.minimum(y12, tf.transpose(y22))

    # compute the areas
    inter_area = tf.maximum((xI2-xI1), 0) * tf.maximum((yI2-yI1), 0) #todo
    inter_area = tf.cast(inter_area, dtype=tf.float32)

    gt_area = (x12 - x11) * (y21 - y11)  # remember: we assume x/y coordinates are inverted
    pred_area = (x22 - x21) * (y22 - y21)
    gt_area = tf.cast(gt_area, dtype=tf.float32)
    pred_area = tf.cast(pred_area, dtype=tf.float32)

    union = (gt_area + tf.transpose(pred_area)) - inter_area
    return inter_area / (union + 1e-05)
    #return inter_area