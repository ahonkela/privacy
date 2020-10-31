# Copyright 2020, The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified to use logistic regression instead of CNN
# and synthetic data instead of MNIST by Antti Honkela, 2019-2020

"""Training a logistic regression model with differentially private SGD optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from absl import logging

import numpy as np
import numpy.random as npr
import tensorflow as tf

import tensorflow.compat.v1 as tf

from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib
from tensorflow_privacy.privacy.optimizers import dp_optimizer


FLAGS = {
  # If True, train with DP-SGD. If False, train with vanilla SGD.
  'dpsgd': True,
  # Learning rate for training
  'learning_rate': 0.05,
  # Ratio of the standard deviation to the clipping norm
  'noise_multiplier': 2.0,
  # Clipping norm
  'l2_norm_clip': 1.0,
  # Batch size
  'batch_size': 64,
  # Number of epochs
  'epochs': 2,
  # Training data size
  'training_data_size': 2000,
  # Test data size
  'test_data_size': 2000,
  # Input dimension
  'input_dimension': 5,
  # Model directory for TensorFlow
  'model_dir': None}


def lr_model_fn(features, labels, mode, params):
  """Model function for a LR."""
  flags = params
  # Define logistic regression model using tf.keras.layers.
  logits = tf.keras.layers.Dense(2).apply(features['x'])

  # Calculate loss as a vector (to support microbatches in DP-SGD).
  vector_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits)
  # Define mean of loss across minibatch (for reporting through tf.Estimator).
  scalar_loss = tf.reduce_mean(input_tensor=vector_loss)

  # Configure the training op (for TRAIN mode).
  if mode == tf.estimator.ModeKeys.TRAIN:

    if flags['dpsgd']:
      # Use DP version of GradientDescentOptimizer. Other optimizers are
      # available in dp_optimizer. Most optimizers inheriting from
      # tf.train.Optimizer should be wrappable in differentially private
      # counterparts by calling dp_optimizer.optimizer_from_args().
      optimizer = dp_optimizer.DPGradientDescentGaussianOptimizer(
          l2_norm_clip=flags['l2_norm_clip'],
          noise_multiplier=flags['noise_multiplier'],
          num_microbatches=None,
          learning_rate=flags['learning_rate'])
      opt_loss = vector_loss
    else:
      optimizer = tf.train.GradientDescentOptimizer(
          learning_rate=flags['learning_rate'])
      opt_loss = scalar_loss

    global_step = tf.train.get_global_step()
    train_op = optimizer.minimize(loss=opt_loss, global_step=global_step)

    # In the following, we pass the mean of the loss (scalar_loss) rather than
    # the vector_loss because tf.estimator requires a scalar loss. This is only
    # used for evaluation and debugging by tf.estimator. The actual loss being
    # minimized is opt_loss defined above and passed to optimizer.minimize().
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=scalar_loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode).
  elif mode == tf.estimator.ModeKeys.EVAL:
    eval_metric_ops = {
        'accuracy':
            tf.metrics.accuracy(
                labels=labels,
                predictions=tf.argmax(input=logits, axis=1))
    }
    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=scalar_loss,
                                      eval_metric_ops=eval_metric_ops)


def generate_data(flags):
  npr.seed(4242)
  N_train = flags['training_data_size']
  N_test = flags['test_data_size']
  N = N_train + N_test
  X0 = npr.randn(N, flags['input_dimension'])
  temp = X0 @ npr.randn(flags['input_dimension'], 1) + npr.randn(N, 1)
  Y0 = np.round(1/(1+np.exp(-temp)))

  train_X = X0[0:N_train, :]
  test_X = X0[N_train:N, :]
  train_Y = Y0[0:N_train, 0]
  test_Y = Y0[N_train:N, 0]
  train_X = np.array(train_X, dtype=np.float32)
  test_X = np.array(test_X, dtype=np.float32)
  train_Y = np.array(train_Y, dtype=np.int32)
  test_Y = np.array(test_Y, dtype=np.int32)
  return train_X, train_Y, test_X, test_Y


def main(flags, data):
  logging.set_verbosity(logging.INFO)
  tf.logging.set_verbosity(tf.logging.ERROR)

  train_data, train_labels, test_data, test_labels = data

  # Instantiate the tf.Estimator.
  lr_classifier = tf.estimator.Estimator(model_fn=lr_model_fn,
                                         model_dir=flags['model_dir'],
                                         params=flags)

  # Create tf.Estimator input functions for the training and test data.
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={'x': train_data},
      y=train_labels,
      batch_size=flags['batch_size'],
      num_epochs=flags['epochs'],
      shuffle=True)
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={'x': test_data},
      y=test_labels,
      num_epochs=1,
      shuffle=False)

  # Training loop.
  steps_per_epoch = flags['training_data_size'] // flags['batch_size'] / 10
  for epoch in range(1, 10*flags['epochs'] + 1):
    start_time = time.time()
    # Train the model for one epoch.
    lr_classifier.train(
        input_fn=train_input_fn, steps=steps_per_epoch)
    end_time = time.time()
    logging.info('Epoch %.1f time in seconds: %.2f', epoch/10, end_time - start_time)

    # Evaluate the model and print results
    eval_results = lr_classifier.evaluate(
        input_fn=eval_input_fn)
    test_accuracy = eval_results['accuracy']
    print('Test accuracy after %.1f epochs is: %.3f' % (epoch/10, test_accuracy))

    # Compute the privacy budget expended.
    if flags['dpsgd']:
      if flags['noise_multiplier'] > 0.0:
        eps, _ = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy(
            flags['training_data_size'], flags['batch_size'], flags['noise_multiplier'], epoch/10, 1e-5)
        print('For delta=1e-5, the current epsilon is: %.2f' % eps)
      else:
        print('Trained with DP-SGD but with zero noise.')
    else:
      print('Trained with vanilla non-private SGD optimizer')


if __name__ == '__main__':
  # Generate training and test data.
  data = generate_data(FLAGS)
  main(FLAGS, data)
