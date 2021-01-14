"""
Training helpers for supervised meta-learning.
"""

import os
import time

import tensorflow as tf

from .reptile import Reptile
from .variables import weight_decay

# pylint: disable=R0913,R0914
def train(sess,
          model,
          train_set,
          test_set,
          save_dir,
          num_shots=4,
          inner_batch_size=None,
          inner_iters=10,
          replacement=False,
          meta_step_size=0.1,
          meta_step_size_final=0.1,
          meta_batch_size=1,
          meta_iters=1000,
          eval_inner_batch_size=None,
          eval_inner_iters=10,
          eval_interval=10,
          weight_decay_rate=1,
          num_train_shots=None,
          time_deadline=None,          
          reptile_fn=Reptile,
          num_eval_samples=100,
          log_fn=print):
    """
    Train a model on a dataset.
    """
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    saver = tf.train.Saver()
    reptile = reptile_fn(sess,
                         pre_step_op=weight_decay(weight_decay_rate))
    accuracy_ph = tf.placeholder(tf.float32, shape=())
    tf.summary.scalar('accuracy', accuracy_ph)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(save_dir, 'train')) #, sess.graph)
    test_writer = tf.summary.FileWriter(os.path.join(save_dir, 'test')) #, sess.graph)
    tf.global_variables_initializer().run()
    sess.run(tf.global_variables_initializer())
    for i in range(meta_iters):
        frac_done = i / meta_iters
        # Linear decay of meta_step_size.
        cur_meta_step_size = frac_done * meta_step_size_final + (1 - frac_done) * meta_step_size
        reptile.train_step(train_set, model.input_ph, model.label_ph, model.minimize_op,
                           num_shots=(num_train_shots or num_shots),
                           inner_batch_size=inner_batch_size, inner_iters=inner_iters,
                           replacement=replacement,
                           meta_step_size=cur_meta_step_size, meta_batch_size=meta_batch_size)
        if i % eval_interval == 0:
            accuracies = []
            for dataset, writer in [(train_set, train_writer), (test_set, test_writer)]:
                total_correct = 0
                for _ in range(num_eval_samples):
                  correct = reptile.evaluate(dataset, model.input_ph, model.label_ph,
                                             model.minimize_op, model.predictions,
                                             num_shots=num_shots,
                                             inner_batch_size=eval_inner_batch_size,
                                             inner_iters=eval_inner_iters, replacement=replacement)
                  total_correct += correct
                acc = float(total_correct) / (2 * num_eval_samples)
                summary = sess.run(merged, feed_dict={
                    accuracy_ph: acc
                })
                writer.add_summary(summary, i)
                writer.flush()
                accuracies.append(acc)
            log_fn('meta iter %d: train=%f test=%f' % (i, accuracies[0], accuracies[1]))
        if i % eval_interval == 0 or i == meta_iters-1:
            saver.save(sess, os.path.join(save_dir, 'model.ckpt'), global_step=i)
        if time_deadline is not None and time.time() > time_deadline:
            break
