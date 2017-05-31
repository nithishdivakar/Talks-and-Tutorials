import tensorflow as tf


def cross_entropy(labels, logits):
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels  = labels, logits = logits)
  return tf.reduce_mean(loss)
