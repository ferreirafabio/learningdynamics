import tensorflow as tf

def create_loss_ops(target_op, output_ops):
    """ ground truth nodes are given by tensor target_op of shape (n_nodes*experience_length, node_output_size) but output_ops
    is a list of graph tuples with shape (n_nodes, exp_len) --> split at the first dimension in order to compute node-wise MSE error
    --> same applies for edges """
    mult = tf.constant([len(output_ops)])
    n_nodes = [tf.shape(output_ops[0].nodes)[0]]
    n_edges = [tf.shape(output_ops[0].edges)[0]]
    node_splits = tf.split(target_op.nodes, num_or_size_splits=tf.tile(n_nodes, mult), axis=0)
    edge_splits = tf.split(target_op.edges, num_or_size_splits=tf.tile(n_edges, mult), axis=0)

    loss_ops = [
        tf.losses.mean_squared_error(output_op.nodes, node_splits[i]) + tf.losses.mean_squared_error(output_op.edges, edge_splits[i])
        for i, output_op in enumerate(output_ops)
    ]
    # todo: might use weighted MSE loss here
    # todo: perhaps include global attributes into loss function

    return loss_ops



