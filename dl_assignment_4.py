from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range


pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10
num_channels = 1 # grayscale

import numpy as np


def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,np.newaxis]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

## problem 1
#
# batch_size = 16
# patch_size = 5
# depth = 16
# num_hidden = 64
#
# graph = tf.Graph()
#
# with graph.as_default():
#     num_steps = 1001
#
#     # Input data.
#     tf_train_dataset = tf.placeholder(
#         tf.float32, shape=(batch_size, image_size, image_size, num_channels))
#     tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
#     tf_valid_dataset = tf.constant(valid_dataset)
#     tf_test_dataset = tf.constant(test_dataset)
#
#     global_step = tf.Variable(0)  # count the number of steps taken.
#     learning_rate = tf.train.exponential_decay(0.05, global_step, num_steps / (np.log(0.01 / 0.05) / np.log(0.86)), 0.86, staircase=True)
#
#     # Variables.
#     layer1_weights = tf.Variable(tf.truncated_normal(
#         [patch_size, patch_size, num_channels, depth], stddev=0.1))
#     layer1_biases = tf.Variable(tf.zeros([depth]))
#     layer2_weights = tf.Variable(tf.truncated_normal(
#         [patch_size, patch_size, depth, depth], stddev=0.1))
#     layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
#     layer3_weights = tf.Variable(tf.truncated_normal(
#         [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
#     layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
#     layer4_weights = tf.Variable(tf.truncated_normal(
#         [num_hidden, num_labels], stddev=0.1))
#     layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
#
#     # pooling
#     def max_pool_2x2(x):
#         return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
#                         strides=[1, 2, 2, 1], padding='SAME')
#     # Model.
#     def model(data):
#         # conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
#         conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
#         hidden = tf.nn.relu(conv + layer1_biases)
#         hidden = max_pool_2x2(hidden)
#         # conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
#         conv = tf.nn.conv2d(hidden, layer2_weights, [1, 1, 1, 1], padding='SAME')
#         hidden = tf.nn.relu(conv + layer2_biases)
#         hidden = max_pool_2x2(hidden)
#         shape = hidden.get_shape().as_list()
#         reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])  # flatten for FC layer.
#         hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
#         return tf.matmul(hidden, layer4_weights) + layer4_biases
#
#
#     # Training computation.
#     logits = model(tf_train_dataset)
#     loss = tf.reduce_mean(
#         tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
#
#     # Optimizer.
#     # optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)
#
#     # Predictions for the training, validation, and test data.
#     train_prediction = tf.nn.softmax(logits)
#     valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
#     test_prediction = tf.nn.softmax(model(tf_test_dataset))
#
#     # num_steps = 1001
#
#     with tf.Session(graph=graph) as session:
#         tf.global_variables_initializer().run()
#         print('Initialized')
#         for step in range(num_steps):
#             offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
#             batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
#             batch_labels = train_labels[offset:(offset + batch_size), :]
#             feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
#             _, l, predictions = session.run(
#                 [optimizer, loss, train_prediction], feed_dict=feed_dict)
#             if (step % 50 == 0):
#                 print('learning rate: ', learning_rate.eval())
#                 print('Minibatch loss at step %d: %f' % (step, l))
#                 print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
#                 print('Validation accuracy: %.1f%%' % accuracy(
#                     valid_prediction.eval(), valid_labels))
#         print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
#

## problem 2
# LeNet5
batch_size = 16
patch_size = 5
c1_depth = 6
c3_depth = 16
f5_depth = 120
f6_depth = 84
graph = tf.Graph()
num_steps = 1001

with graph.as_default():
    # Input data.
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # dropout
    keep_prob = tf.placeholder(tf.float32)

    # learning rate decay
    global_step = tf.Variable(0)  # count the number of steps taken.
    initial_learning_rate = 0.05
    final_learning_rate = 0.01
    decay_rate = 0.96
    learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_rate=decay_rate,
                                               decay_steps=num_steps / (
                                               np.log(final_learning_rate / initial_learning_rate) / np.log(
                                                   decay_rate)))

    # Variables.
    c1_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, num_channels, c1_depth], stddev=0.1))
    # c1_weights = tf.get_variable('c1_weights', [patch_size, patch_size, num_channels, c1_depth], initializer=tf.contrib.layers.xavier_initializer())
    c1_biases = tf.Variable(tf.zeros([c1_depth]))
    c3_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, c1_depth, c3_depth], stddev=0.1))
    # c3_weights = tf.get_variable('c3_weights', [patch_size, patch_size, c1_depth, c3_depth], initializer=tf.contrib.layers.xavier_initializer())
    c3_biases = tf.Variable(tf.constant(1.0, shape=[c3_depth]))
    f5_weights = tf.Variable(tf.truncated_normal(
        [image_size // 4 * image_size // 4 * c3_depth, f5_depth], stddev=0.1))
    # f5_weights = tf.get_variable('f5_weights', [image_size // 4 * image_size // 4 * c3_depth, f5_depth], initializer=tf.contrib.layers.xavier_initializer())
    f5_biases = tf.Variable(tf.constant(1.0, shape=[f5_depth]))
    f6_weights = tf.Variable(tf.truncated_normal(
        [f5_depth, f6_depth], stddev=0.1))
    # f6_weights = tf.get_variable('f6_weights', [f5_depth, f6_depth], initializer=tf.contrib.layers.xavier_initializer())
    f6_biases = tf.Variable(tf.constant(1.0, shape=[f6_depth]))
    readout_weights = tf.Variable(tf.truncated_normal(
        [f6_depth, num_labels], stddev=0.1))
    # readout_weights = tf.get_variable('readout_weights', [f6_depth, num_labels], initializer=tf.contrib.layers.xavier_initializer())
    readout_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

    # pooling
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    # Model.
    def model(data):
        # conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
        conv = tf.nn.conv2d(data, c1_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + c1_biases)
        hidden = max_pool_2x2(hidden)
        # conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
        conv = tf.nn.conv2d(hidden, c3_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + c3_biases)
        hidden = max_pool_2x2(hidden)
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, f5_weights) + f5_biases)
        hidden = tf.nn.relu(tf.matmul(hidden, f6_weights) + f6_biases)
        return tf.matmul(hidden, readout_weights) + readout_biases

    # # evaluation Model.
    # def eval_model(data):
    #     # conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
    #     conv = tf.nn.conv2d(data, c1_weights, [1, 1, 1, 1], padding='SAME')
    #     hidden = tf.nn.relu(conv + c1_biases)
    #     hidden = max_pool_2x2(hidden)
    #     # conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
    #     conv = tf.nn.conv2d(hidden, c3_weights, [1, 1, 1, 1], padding='SAME')
    #     hidden = tf.nn.relu(conv + c3_biases)
    #     hidden = max_pool_2x2(hidden)
    #     shape = hidden.get_shape().as_list()
    #     reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
    #     hidden = tf.nn.relu(tf.matmul(reshape, f5_weights) + f5_biases)
    #     hidden = tf.nn.relu(tf.matmul(hidden, f6_weights) + f6_biases)
    #     return tf.matmul(hidden, readout_weights) + readout_biases

    # Training computation.
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)
    # optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print('Initialized')
        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
            _, l, predictions = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 50 == 0):
                print('learning rate:', learning_rate.eval())
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                print('Validation accuracy: %.1f%%' % accuracy(
                    valid_prediction.eval(), valid_labels))
        print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
