import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

EPSILON = 0.00001

def tensor_layer_norm(x, state_name):
    x_shape = x.get_shape()
    dims = x_shape.ndims
    params_shape = x_shape[-1:]
    if dims == 4:
        m, v = tf.nn.moments(x, [1,2,3], keep_dims=True)
    elif dims == 5:
        m, v = tf.nn.moments(x, [1,2,3,4], keep_dims=True)
    else:
        raise ValueError('input tensor for layer normalization must be rank 4 or 5.')
    b = tf.get_variable(state_name+'b',initializer=tf.zeros(params_shape))
    s = tf.get_variable(state_name+'s',initializer=tf.ones(params_shape))
    x = tf.cast(x, tf.float32)
    m = tf.cast(m, tf.float32)
    v = tf.cast(v, tf.float32)
    b = tf.cast(b, tf.float32)
    s = tf.cast(s, tf.float32)
    x_tln = tf.nn.batch_normalization(x, m, v, b, s, EPSILON)
    return x_tln
