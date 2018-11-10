import tensorflow as tf

def create_loss_ops(config, target_op, output_ops):
    """ ground truth nodes are given by tensor target_op of shape (n_nodes*experience_length, node_output_size) but output_ops
    is a list of graph tuples with shape (n_nodes, exp_len) --> split at the first dimension in order to compute node-wise MSE error
    --> same applies for edges """
    mult = tf.constant([len(output_ops)])
    n_nodes = [tf.shape(output_ops[0].nodes)[0]]
    n_edges = [tf.shape(output_ops[0].edges)[0]]
    node_splits = tf.split(target_op.nodes, num_or_size_splits=tf.tile(n_nodes, mult), axis=0, name='a')
    edge_splits = tf.split(target_op.edges, num_or_size_splits=tf.tile(n_edges, mult), axis=0)

    """ if object seg data is only used for init, the ground truth features in the rest of the sequence are static except position 
    --> in this case compute loss only over the position since image prediction is infeasible """
    if not config.use_object_seg_data_only_for_init:
        loss_ops = [
            tf.losses.mean_squared_error(output_op.nodes, node_splits[i]) +
            tf.losses.mean_squared_error(output_op.edges, edge_splits[i])
            for i, output_op in enumerate(output_ops)
        ]
    else:
        loss_ops = [
            # compute loss of the nodes only over velocity and position
            # todo: should container be removed entirely?
            tf.losses.mean_squared_error(output_op.nodes, node_splits[i][:,-3:]) +
            tf.losses.mean_squared_error(output_op.edges, edge_splits[i])
            for i, output_op in enumerate(output_ops)
        ]

    # todo: might use weighted MSE loss here
    # todo: perhaps include global attributes into loss function

    return loss_ops, node_splits



