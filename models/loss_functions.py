import tensorflow as tf
from utils.utils import get_correct_image_shape


def create_loss_ops(config, target_op, output_ops):
    """ ground truth nodes are given by tensor target_op of shape (n_nodes*experience_length, node_output_size) but output_ops
    is a list of graph tuples with shape (n_nodes, node_output_size) --> split at the first dimension in order to compute node-wise MSE error
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

    targ = target_op

    if config.normalize_data:
        img_scale = 10
        non_visual_scale = 1
    else:
        img_scale = 1
        non_visual_scale = 1

    if config.loss_type == 'mse':
        for i, output_op in enumerate(output_ops):
            ''' checks whether the padding_flag is 1 or 0 --> if 1, return inf loss (later removed)'''
            condition = tf.equal(target_op.globals[i, 0], tf.constant(1.0))

            """ VISUAL LOSS """
            loss_visual_mse_nodes = tf.cond(condition, lambda: float("inf"),
                                            lambda: img_scale * tf.losses.mean_squared_error(
                                                labels=target_node_splits[i][:, :-6],
                                                predictions=output_op.nodes[:, :-6],
                                                weights=0.5)
                                            )

            """ NONVISUAL LOSS (50% weight) """
            loss_nonvisual_mse_edges = tf.cond(condition, lambda: float("inf"),
                                               lambda: non_visual_scale * tf.losses.mean_squared_error(
                                                   labels=target_edge_splits[i],
                                                   predictions=output_op.edges,
                                                   weights=(1/6))
                                               )
            loss_nonvisual_mse_nodes_pos = tf.cond(condition, lambda: float("inf"),
                                                   lambda: non_visual_scale * tf.losses.mean_squared_error(
                                                       labels=target_node_splits[i][:, -3:],
                                                       predictions=output_op.nodes[:, -3:],
                                                       weights=(1/6))
                                                   )
            loss_nonvisual_mse_nodes_vel = tf.cond(condition, lambda: float("inf"),
                                                   lambda: tf.losses.mean_squared_error(
                                                       labels=target_node_splits[i][:, -6:-3:],
                                                       predictions=output_op.nodes[:, -6:-3:],
                                                       weights=(1/6))
                                                   )

            loss_ops_img.append(loss_visual_mse_nodes)
            loss_ops_velocity.append(loss_nonvisual_mse_nodes_vel)
            loss_ops_position.append(loss_nonvisual_mse_nodes_pos)
            loss_ops_distance.append(loss_nonvisual_mse_edges)

            total_loss_ops.append(loss_visual_mse_nodes + loss_nonvisual_mse_edges + loss_nonvisual_mse_nodes_vel + loss_nonvisual_mse_nodes_pos)

    elif config.loss_type == 'cross_entropy_seg_only':
        for i, output_op in enumerate(output_ops):
            ''' checks whether the padding_flag is 1 or 0 --> if 1, return inf loss (later removed)'''
            condition = tf.equal(target_op.globals[i, 0], tf.constant(1.0))

            """ VISUAL LOSS """
            predicted_node_reshaped = _transform_into_images(config, output_op.nodes, img_type="seg")  # returns shape (?, 120,160,1)
            target_node_reshaped = _transform_into_images(config, target_node_splits[i], img_type="all")  # returns shape (?, 120,160,7)
            segmentation_data_predicted = predicted_node_reshaped[:, :, :, 0]  # --> transform into (?,120,160)
            segmentation_data_gt = target_node_reshaped[:, :, :, 3]  # --> transform into (?,120,160)

            logits = tf.reshape(segmentation_data_predicted, [-1,
                                                              segmentation_data_predicted.get_shape()[1] *
                                                              segmentation_data_predicted.get_shape()[2]])
            labels = tf.reshape(segmentation_data_gt, [-1,
                                                       segmentation_data_gt.get_shape()[1] *
                                                       segmentation_data_gt.get_shape()[2]])

            ones = tf.ones_like(labels)
            comparison = tf.equal(labels, tf.constant(1.0))
            pos_weight = tf.where(comparison, ones*0.8, ones)  # decrease false negative count

            loss_visual_mse_nodes = tf.cond(condition, lambda: float("inf"),
                                                lambda: img_scale * 0.5 * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
                                                        targets=labels,
                                                        logits=logits,
                                                        pos_weight=pos_weight))
                                            )

            tf.losses.add_loss(loss_visual_mse_nodes)

            loss_visual_iou_seg = 0.0  # no iou loss computed

            """ NONVISUAL LOSS (50% weight) """
            loss_nonvisual_mse_edges = tf.cond(condition, lambda: float("inf"),
                                               lambda: non_visual_scale * tf.losses.mean_squared_error(
                                                   labels=target_edge_splits[i],
                                                   predictions=output_op.edges,
                                                   weights=0.1)
                                               )
            loss_nonvisual_mse_nodes_pos = tf.cond(condition, lambda: float("inf"),
                                                   lambda: non_visual_scale * tf.losses.mean_squared_error(
                                                       labels=target_node_splits[i][:, -3:],
                                                       predictions=output_op.nodes[:, -3:],
                                                       weights=0.3)
                                                   )
            loss_nonvisual_mse_nodes_vel = tf.cond(condition, lambda: float("inf"),
                                                   lambda: tf.losses.mean_squared_error(
                                                       labels=target_node_splits[i][:, -6:-3:],
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

    elif config.loss_type == 'mse_seg_only':
        for i, output_op in enumerate(output_ops):
            ''' checks whether the padding_flag is 1 or 0 --> if 1, return inf loss (later removed)'''
            condition = tf.equal(target_op.globals[i, 0], tf.constant(1.0))

            """ VISUAL LOSS """
            predicted_node_reshaped = _transform_into_images(config, output_op.nodes,
                                                             img_type="seg")  # returns shape (?, 120,160,1)
            target_node_reshaped = _transform_into_images(config, target_node_splits[i],
                                                          img_type="all")  # returns shape (?, 120,160,7)
            segmentation_data_predicted = predicted_node_reshaped[:, :, :, 0]  # --> transform into (?,120,160)
            segmentation_data_gt = target_node_reshaped[:, :, :, 3]  # --> transform into (?,120,160)

            logits = tf.reshape(segmentation_data_predicted, [-1,
                                                              segmentation_data_predicted.get_shape()[1] *
                                                              segmentation_data_predicted.get_shape()[2]])
            labels = tf.reshape(segmentation_data_gt, [-1,
                                                       segmentation_data_gt.get_shape()[1] *
                                                       segmentation_data_gt.get_shape()[2]])

            loss_visual_mse_nodes = tf.cond(condition, lambda: float("inf"),
                                            lambda: img_scale * tf.losses.mean_squared_error(
                                                labels=labels,
                                                predictions=logits,
                                                weights=0.5)
                                            )

            loss_visual_iou_seg = 0.0  # no iou loss computed

            """ NONVISUAL LOSS (50% weight) """
            loss_nonvisual_mse_edges = tf.cond(condition, lambda: float("inf"),
                                               lambda: non_visual_scale * tf.losses.mean_squared_error(
                                                   labels=target_edge_splits[i],
                                                   predictions=output_op.edges,
                                                   weights=(1 / 6))
                                               )
            loss_nonvisual_mse_nodes_pos = tf.cond(condition, lambda: float("inf"),
                                                   lambda: non_visual_scale * tf.losses.mean_squared_error(
                                                       labels=target_node_splits[i][:, -3:],
                                                       predictions=output_op.nodes[:, -3:],
                                                       weights=(1 / 6))
                                                   )
            loss_nonvisual_mse_nodes_vel = tf.cond(condition, lambda: float("inf"),
                                                   lambda: tf.losses.mean_squared_error(
                                                       labels=target_node_splits[i][:,-6:-3:],
                                                       predictions=output_op.nodes[:, -6:-3:],
                                                       weights=(1 / 6))
                                                   )

            loss_ops_img.append(loss_visual_mse_nodes)
            loss_ops_img_iou.append(loss_visual_iou_seg)
            loss_ops_velocity.append(loss_nonvisual_mse_nodes_vel)
            loss_ops_position.append(loss_nonvisual_mse_nodes_pos)
            loss_ops_distance.append(loss_nonvisual_mse_edges)

            #print("---- image loss only ----")
            #total_loss_ops.append(loss_visual_mse_nodes)
            total_loss_ops.append(loss_visual_mse_nodes + loss_nonvisual_mse_edges + loss_nonvisual_mse_nodes_vel + loss_nonvisual_mse_nodes_pos)

    elif config.loss_type == 'mse_iou':
        for i, output_op in enumerate(output_ops):
            ''' checks whether the padding_flag is 1 or 0 --> if 1, return inf loss (later removed)'''
            condition = tf.equal(target_op.globals[i, 0], tf.constant(1.0))

            """ VISUAL LOSS """
            loss_visual_mse_nodes = tf.cond(condition, lambda: float("inf"),
                                            lambda: img_scale * tf.losses.mean_squared_error(
                                                labels=target_node_splits[i][:, :-6],
                                                predictions=output_op.nodes[:, :-6],
                                                weights=0.4)
                                            )

            predicted_node_reshaped = _transform_into_images(config, output_op.nodes)
            target_node_reshaped = _transform_into_images(config, target_node_splits[i])
            segmentation_data_predicted = predicted_node_reshaped[:, :, :, 3]
            segmentation_data_gt = target_node_reshaped[:, :, :, 3]

            loss_visual_iou_seg = tf.cond(condition, lambda: float("inf"),
                                            lambda: 0.1 *_intersection_over_union(
                                                image_gt=segmentation_data_gt,
                                                image_pred=segmentation_data_predicted,
                                                config=config)
                                            )

            """ NONVISUAL LOSS (50% weight) """
            loss_nonvisual_mse_edges = tf.cond(condition, lambda: float("inf"),
                                               lambda: non_visual_scale * tf.losses.mean_squared_error(
                                                   labels=target_edge_splits[i],
                                                   predictions=output_op.edges,
                                                   weights=(1/6))
                                               )
            loss_nonvisual_mse_nodes_pos = tf.cond(condition, lambda: float("inf"),
                                                   lambda: non_visual_scale * tf.losses.mean_squared_error(
                                                       labels=target_node_splits[i][:, -3:],
                                                       predictions=output_op.nodes[:, -3:],
                                                       weights=(1/6))
                                                   )
            loss_nonvisual_mse_nodes_vel = tf.cond(condition, lambda: float("inf"),
                                                   lambda: tf.losses.mean_squared_error(
                                                       labels=target_node_splits[i][:, -6:-3:],
                                                       predictions=output_op.nodes[:, -6:-3:],
                                                       weights=(1/6))
                                                   )

            loss_ops_img.append(loss_visual_mse_nodes)
            loss_ops_img_iou.append(loss_visual_iou_seg)
            loss_ops_velocity.append(loss_nonvisual_mse_nodes_vel)
            loss_ops_position.append(loss_nonvisual_mse_nodes_pos)
            loss_ops_distance.append(loss_nonvisual_mse_edges)

            total_loss_ops.append(loss_visual_mse_nodes + loss_visual_iou_seg + loss_nonvisual_mse_edges + loss_nonvisual_mse_nodes_vel + loss_nonvisual_mse_nodes_pos)

    elif config.loss_type == 'mse_gdl':
        # todo: change to new condition/padding_flag version
        for i, output_op in enumerate(output_ops):
            """ VISUAL LOSS (50% weight) """
            loss_visual_mse_nodes = img_scale * tf.losses.mean_squared_error(labels=target_node_splits[i][:, :-6], predictions=output_op.nodes[:, :-6], weights=0.25)

            # pre-processing step for gdl function call
            predicted_node_reshaped = _transform_into_images(config, output_op.nodes)
            target_node_reshaped = _transform_into_images(config, target_node_splits[i])
            loss_visual_gdl_nodes = 0.25 * gradient_difference_loss(true=target_node_reshaped, pred=predicted_node_reshaped)

            """ NONVISUAL LOSS (50% weight) """
            loss_nonvisual_mse_edges = tf.losses.mean_squared_error(labels=target_edge_splits[i], predictions=output_op.edges, weights=(1/6))
            loss_nonvisual_mse_nodes_pos = tf.losses.mean_squared_error(labels=target_node_splits[i][:, -3:], predictions=output_op.nodes[:, -3:], weights=(1/6))
            loss_nonvisual_mse_nodes_vel = tf.losses.mean_squared_error(labels=target_node_splits[i][:, -6:-3:], predictions=output_op.nodes[:, -6:-3:], weights=(1/6))

            loss_ops_img.append(loss_visual_gdl_nodes + loss_visual_mse_nodes)
            loss_ops_velocity.append(loss_nonvisual_mse_nodes_vel)
            loss_ops_position.append(loss_nonvisual_mse_nodes_pos)
            loss_ops_distance.append(loss_nonvisual_mse_edges)

            total_loss_ops.append(loss_visual_mse_nodes + loss_visual_gdl_nodes + loss_nonvisual_mse_edges + loss_nonvisual_mse_nodes_pos + loss_nonvisual_mse_nodes_vel)

    else:
        raise ValueError("loss type must be in [\"mse\", \"mse_gdl\", \"mse_iou\", \"mse_seg_only\", \"cross_entropy_seg_only\" but is {}".format(config.loss_type))

    #l2_loss = tf.losses.get_regularization_loss()
    #total_loss_ops += l2_loss

    return total_loss_ops, loss_ops_img, loss_ops_img_iou, loss_ops_velocity, loss_ops_position, loss_ops_distance, targ


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


def _transform_into_images(config, data, img_type="all"):
    """ reshapes data (shape: (batch_size, feature_length)) into the required image shape with an
    additional batch_dimension, e.g. (1,120,160,7) """
    data_shape = get_correct_image_shape(config, get_type=img_type)
    data = data[:, :-6]
    data = tf.reshape(data, [-1, *data_shape])
    return data


def _intersection_over_union(image_gt, image_pred, config):
    """
    :param image_gt: the ground truth segmentation image, expects a tensor of shape (120,160,1)
    :param image_gt: the predicted image, expects a tensor of shape (120,160,1)
    :return: the intersection over union. The union is computed for the bounding boxes represented by the min and max values of the segments
    """
    if config.normalize_data:
        # when normalization is activated, background pixels (smallest values) are transformed from 0 to another number --> transformation
        # is linear therefore take smallest occurring value
        background_pixel_value = tf.cast(tf.reduce_min(image_gt), dtype=tf.float32)
    else:
        background_pixel_value = tf.cast(0.0, dtype=tf.float32)

    mask_pred = image_pred > background_pixel_value
    coordinates_pred = tf.where(mask_pred)

    mask_gt = tf.cast(image_gt, dtype=tf.float32) > background_pixel_value
    coordinates_gt = tf.where(mask_gt)

    # the following is a test line; should yield 1 (IoU) or 0 (1-IoU)
    # coordinates_pred = coordinates_gt

    xy1_max = tf.reduce_max(coordinates_pred, axis=0)
    x12 = xy1_max[1]  # numpy coordinates are (y,x) instead of (x,y), origin of coordinate system is top left
    y12 = xy1_max[0]

    xy1_min = tf.reduce_min(coordinates_pred, axis=0)
    x11 = xy1_min[1]
    y11 = xy1_min[0]

    xy2_max = tf.reduce_max(coordinates_gt, axis=0)
    x22 = xy2_max[1]
    y22 = xy2_max[0]

    xy2_min = tf.reduce_min(coordinates_gt, axis=0)
    x21 = xy2_min[1]
    y21 = xy2_min[0]

    # determine the (x,y) coordinates of the intersection rectangle
    xI1 = tf.maximum(x11, x21)
    xI2 = tf.minimum(x12, x22)

    yI1 = tf.maximum(y11, y21)
    yI2 = tf.minimum(y12, y22)

    # compute the areas
    inter_area = tf.maximum(xI2-xI1, 0) * tf.maximum(yI2-yI1, 0)
    inter_area = tf.cast(inter_area, dtype=tf.float32)

    pred_area = (x12 - x11) * (y12 - y11)
    gt_area = (x22 - x21) * (y22 - y21)
    gt_area = tf.cast(gt_area, dtype=tf.float32)
    pred_area = tf.cast(pred_area, dtype=tf.float32)

    # take the negative of IoU to get IoU loss
    union = gt_area + pred_area - inter_area
    return 1-(inter_area / (union + 1e-05))
