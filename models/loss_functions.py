import tensorflow as tf

def create_loss_ops(target_op, output_ops):
    #todo: might use weighted MSE loss here
    loss_ops = [
      tf.losses.mean_squared_error(target_op.nodes, output_op.nodes) + tf.losses.mean_squared_error(target_op.edges, output_op.edges)
        for output_op in output_ops
               ]
    return loss_ops

