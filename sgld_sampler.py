import tensorflow as tf


@tf.function
def sgld_sample(E, x_initial, num_steps, step_size, var, clip_thresh=tf.constant(0.01), constrain_results=False):
    """
    Do SGLD (stochastic gradient Langevin dynamics) sampling step
    :param E: Energy function
    :param x_initial: initial sample position, of shape (batch_size, ndims)
    :param num_steps: number of sampling steps
    :param step_size: step size used in gradient part
    :param var: variance for isotropic Gaussian used in update
    :param clip_thresh: threshold for gradient clipping; prevents energy gradients from growing too large
    :param constrain_results: Constrain results between 0 and 1
    :return: new sample
    """
    x_k = x_initial
    for _ in range(num_steps):
        with tf.GradientTape() as g:
            g.watch(x_k)
            energy = tf.math.reduce_sum(E(x_k, training=False))
        gradient = g.gradient(energy, x_k)
        dE_dx = tf.clip_by_value(gradient, -clip_thresh, clip_thresh)
        x_k = x_k - (step_size / 2) * dE_dx + tf.random.normal(x_k.shape, mean=0.0, stddev=tf.math.sqrt(var))
    if constrain_results:
        x_k = tf.clip_by_value(x_k, 0.0, 1.0)
    return x_k


@tf.function
def sgld_sample_diag(E, x_initial, num_steps, step_size, var, clip_thresh=tf.constant(0.01), constrain_results=False):
    """
    Do SGLD (stochastic gradient Langevin dynamics) sampling step, record average gradient magnitude
    :param E: Energy function
    :param x_initial: initial sample position, of shape (batch_size, ndims)
    :param num_steps: number of sampling steps
    :param step_size: step size used in gradient part
    :param var: variance for isotropic Gaussian used in update
    :param clip_thresh: threshold for gradient clipping; prevents energy gradients from growing too large
    :param constrain_results: Constrain results between 0 and 1
    :return: new sample, avg. gradient magnitude
    """
    x_k = x_initial
    r_s = 0.0
    for _ in range(num_steps):
        with tf.GradientTape() as g:
            g.watch(x_k)
            energy = tf.math.reduce_sum(E(x_k, training=False))
        gradient = g.gradient(energy, x_k)
        dE_dx = tf.clip_by_value(gradient, -clip_thresh, clip_thresh)
        r_s += tf.reduce_mean(tf.linalg.norm(dE_dx, axis=1))
        x_k = x_k - (step_size / 2) * dE_dx + tf.random.normal(x_k.shape, mean=0.0, stddev=tf.math.sqrt(var))
    if constrain_results:
        x_k = tf.clip_by_value(x_k, 0.0, 1.0)
    return x_k, r_s / tf.cast(num_steps, tf.float32)
