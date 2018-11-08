import tensorflow as tf

def create_loss_ops(target_op, output_ops):
    #todo: might use weighted MSE loss here
    loss_ops = [
      tf.losses.mean_squared_error(target_op.nodes, output_op.nodes) for output_op in output_ops]
    return loss_ops
    # loss = 0.0
    # for output_op in output_ops:
    #     loss += tf.losses.mean_squared_error(target_op.nodes, output_op.nodes)
    # return loss

