"""
Models for supervised startup success prediction.
"""

from functools import partial

import numpy as np
import tensorflow as tf

DEFAULT_OPTIMIZER = partial(tf.train.AdamOptimizer, beta1=0.0)

# pylint: disable=R0903
class NNClassifier:
    """
    A neural network binary classifier.
    """
    def __init__(self, num_classes, n_features, n_layers=1,
                 hidden_size=32, optimizer=DEFAULT_OPTIMIZER, **optim_kwargs):
        self.input_ph = tf.placeholder(tf.float32, shape=(None, n_features))
        out = self.input_ph

        h_prev = n_features
        for i in range(n_layers):
            w = tf.get_variable('w_{}'.format(i), shape=[h_prev, hidden_size],
                                initializer=tf.random_normal_initializer(
                                    0.0, np.sqrt(2.0 / (h_prev + hidden_size))))
            bias = tf.get_variable('b_{}'.format(i), shape=hidden_size,
                                   initializer=tf.constant_initializer(0.0))
            out = apply_layer(out, w, bias)
            # out = tf.layers.dense(out, hidden_size)
            out = tf.nn.relu(out)
            h_prev = hidden_size

        w = tf.get_variable('w_final', shape=[h_prev, num_classes],
                            initializer=tf.random_normal_initializer(
                                0.0, np.sqrt(2.0 / (h_prev + num_classes))))
        bias = tf.get_variable('b_final', shape=num_classes,
                               initializer=tf.constant_initializer(0.0))
        self.logits = apply_layer(out, w, bias)
        self.label_ph = tf.placeholder(tf.int32, shape=(None,))
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.label_ph, logits=self.logits)
        self.predictions = tf.argmax(self.logits, axis=-1)
        self.minimize_op = optimizer(**optim_kwargs).minimize(self.loss)


def apply_layer(a, w, bias):
  a_prev = a
  a = tf.matmul(a, w)
  a += bias
  return a


class HyperNNClassifier:
    """
    A neural network binary classifier.
    """
    def __init__(self, num_classes, n_features, n_layers=1, task_descr_dim=50,
                 hidden_size=32, optimizer=DEFAULT_OPTIMIZER, **optim_kwargs):
        self.input_ph = tf.placeholder(tf.float32, shape=(None, n_features))
        input_, task_descr = tf.split(
            self.input_ph, [n_features - task_descr_dim, task_descr_dim], axis=1)

        task_descr = tf.reshape(task_descr[0], [1, task_descr_dim])

        out = input_
        h_prev = n_features - task_descr_dim
        for i in range(n_layers):
            w, bias = generate_params(task_descr, h_prev, hidden_size)
            out = apply_layer(out, w, bias)
            out = tf.nn.relu(out)
            h_prev = hidden_size

        w, bias = generate_params(task_descr, h_prev, num_classes)
        self.logits = tf.matmul(out, w) + bias
        self.label_ph = tf.placeholder(tf.int32, shape=(None,))
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.label_ph, logits=self.logits)
        self.predictions = tf.argmax(self.logits, axis=-1)
        self.minimize_op = optimizer(**optim_kwargs).minimize(self.loss)


def generate_params(input_, in_dim, out_dim, n_layers=2, hidden_size=10):
    n_params = in_dim * out_dim + out_dim
    out = input_
    for _ in range(n_layers):
        out = tf.layers.dense(out, hidden_size)
        out = tf.math.tanh(out)
    params = tf.layers.dense(out, n_params)
    weights, bias = tf.split(params, [in_dim*out_dim, out_dim], axis=1)
    weights = tf.reshape(weights, [in_dim, out_dim])
    bias = tf.reshape(bias, [out_dim])
    return weights, bias


class ConstantPredictionClassifier:
    """Always predict 0 class.
    """
    def __init__(self, num_classes, n_features, optimizer=DEFAULT_OPTIMIZER, **optim_kwargs):
        self.input_ph = tf.placeholder(tf.float32, shape=(None, n_features))
        out = self.input_ph
        self.logits = tf.layers.dense(out, num_classes)
        self.label_ph = tf.placeholder(tf.int32, shape=(None,))
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_ph,
                                                                   logits=self.logits)
        self.predictions = tf.argmax(self.logits, axis=-1)
        self.minimize_op = optimizer(**optim_kwargs).minimize(self.loss)

        self.predictions = tf.zeros_like(self.predictions)
