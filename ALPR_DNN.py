# Building an Automatic Licence Plate Recognition (ALPR) system using Deep CNN (Convolution Neural Network)
# to be deployed on mobile robots.
# Developed by Nishant Gadhvi, Graduate Researcher, Lamar University.

# Import necessary packages.
import tensorflow as tf
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time

# Training Parameters.
learning_rate = 0.001
num_steps = 1000        # Number of iterations to be performed.
batch_size = 128

# Network Parameters.



# CNN Model Architecture.

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 200, 40, 3])

  # Convolutional Layer Sequence #1
  conv1_1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=16,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1_1
  pool1_1 = tf.layers.max_pooling2d(inputs=conv1_1, pool_size=[2, 2], strides=2)

  conv1_2 = tf.layers.conv2d(
      inputs=pool1_1,
      filters=16,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1_2
  pool1_2 = tf.layers.max_pooling2d(inputs=conv1_2, pool_size=[2, 2], strides=2)

  # Convolutional Layer Sequence #2 and Pooling Layer sequence #2
  conv2_1 = tf.layers.conv2d(
      inputs=pool1_2,
      filters=32,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
  pool2_1 = tf.layers.max_pooling2d(inputs=conv2_1, pool_size=[2, 2], strides=2)

  conv2_2 = tf.layers.conv2d(
      inputs=pool2_1,
      filters=32,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
  pool2_2 = tf.layers.max_pooling2d(inputs=conv2_2, pool_size=[2, 2], strides=2)


  # Convolutional Layer sequence #3 and Pooling Layer sequence #3
  conv3_1 = tf.layers.conv2d(
      inputs=pool2_2,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
  pool3_1 = tf.layers.max_pooling2d(inputs=conv3_1, pool_size=[2, 2], strides=2)

  conv3_2 = tf.layers.conv2d(
      inputs=pool3_1,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
  pool3_2 = tf.layers.max_pooling2d(inputs=conv3_2, pool_size=[2, 2], strides=2)

  # Dense Layer
  pool3_flat = tf.reshape(pool3, [-1, 7 * 7 * 64])
  dense = tf.layers.dense(inputs=pool3_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)