
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle


# pickle_file = 'notMNIST.pickle'
pickle_file = 'notMNIST_santinized.pickle'
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


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size**2)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:, np.newaxis]).astype(np.float32)
    return dataset, labels


train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


## problem 1
# without regularization.
# train %82.8%
# CV %82.3%
# test %89.0%

# # With gradient descent training, even this much data is prohibitive.
# # Subset the training data for faster turnaround.
# train_subset = 10000
#
# graph = tf.Graph()
# with graph.as_default():
#
#     # Input data.
#     # Load the training, validation and test data into constants that are
#     # attached to the graph.
#     tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
#     tf_train_labels = tf.constant(train_labels[:train_subset])
#     tf_valid_dataset = tf.constant(valid_dataset)
#     tf_test_dataset = tf.constant(test_dataset)
#
#     # Variables.
#     # These are the parameters that we are going to be training. The weight
#     # matrix will be initialized using random values following a (truncated)
#     # normal distribution. The biases get initialized to zero.
#     weights = tf.Variable(
#       tf.truncated_normal([image_size * image_size, num_labels]))
#     biases = tf.Variable(tf.zeros([num_labels]))
#
#     # Training computation.
#     # We multiply the inputs with the weight matrix, and add biases. We compute
#     # the softmax and cross-entropy (it's one operation in TensorFlow, because
#     # it's very common, and it can be optimized). We take the average of this
#     # cross-entropy across all training examples: that's our loss.
#     logits = tf.matmul(tf_train_dataset, weights) + biases # the model ouputs before softmax called logits.
#     loss = tf.reduce_mean(
#       tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
#
#     # Optimizer.
#     # We are going to find the minimum of this loss using gradient descent.
#     optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss) # optimizer is the same thing to solver in sklearn
#
#     # Predictions for the training, validation, and test data.
#     # These are not part of training, but merely here so that we can report
#     # accuracy figures as we train.
#     train_prediction = tf.nn.softmax(logits)
#     valid_prediction = tf.nn.softmax(
#       tf.matmul(tf_valid_dataset, weights) + biases)
#     test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
#
#
#     num_steps = 801
#
#
#     def accuracy(predictions, labels):
#       return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))# along axis=1 means by row.
#               / predictions.shape[0])
#
#
#     with tf.Session(graph=graph) as session:
#       # This is a one-time operation which ensures the parameters get initialized as
#       # we described in the graph: random weights for the matrix, zeros for the
#       # biases.
#       tf.global_variables_initializer().run()
#       print('Initialized')
#       for step in range(num_steps):
#         # Run the computations. We tell .run() that we want to run the optimizer,
#         # and get the loss value and the training predictions returned as numpy
#         # arrays.
#         _, l, predictions = session.run([optimizer, loss, train_prediction])
#         if (step % 100 == 0):
#           print('Loss at step %d: %f' % (step, l))
#           print('Training accuracy: %.1f%%' % accuracy(
#             predictions, train_labels[:train_subset, :]))
#           # Calling .eval() on valid_prediction is basically like calling run(), but
#           # just to get that one numpy array. Note that it recomputes all its graph
#           # dependencies.
#           print('Validation accuracy: %.1f%%' % accuracy(
#             valid_prediction.eval(), valid_labels))
#       print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
#
#
# grid_lambda = np.array([1.0, 0.3, 0.1, 0.03, 0.01])
# train_subset = 10000
#
# for regular_lambda in grid_lambda:
#     print('\nregularization coefficient: ', regular_lambda)
#     graph = tf.Graph()
#     with graph.as_default():
#
#         # Input data.
#         # Load the training, validation and test data into constants that are
#         # attached to the graph.
#         tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
#         tf_train_labels = tf.constant(train_labels[:train_subset])
#         tf_valid_dataset = tf.constant(valid_dataset)
#         tf_test_dataset = tf.constant(test_dataset)
#
#         # Variables.
#         # These are the parameters that we are going to be training. The weight
#         # matrix will be initialized using random values following a (truncated)
#         # normal distribution. The biases get initialized to zero.
#         weights = tf.Variable(
#             tf.truncated_normal([image_size * image_size, num_labels]))
#         biases = tf.Variable(tf.zeros([num_labels]))
#
#         # Training computation.
#         # We multiply the inputs with the weight matrix, and add biases. We compute
#         # the softmax and cross-entropy (it's one operation in TensorFlow, because
#         # it's very common, and it can be optimized). We take the average of this
#         # cross-entropy across all training examples: that's our loss.
#         logits = tf.matmul(tf_train_dataset, weights) + biases  # the model ouputs before softmax called logits.
#         loss = tf.reduce_mean(
#             tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)
#             + regular_lambda*tf.nn.l2_loss(weights))
#
#         # Optimizer.
#         # We are going to find the minimum of this loss using gradient descent.
#         optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(
#             loss)  # optimizer is the same thing to solver in sklearn
#
#         # Predictions for the training, validation, and test data.
#         # These are not part of training, but merely here so that we can report
#         # accuracy figures as we train.
#         train_prediction = tf.nn.softmax(logits)
#         valid_prediction = tf.nn.softmax(
#             tf.matmul(tf_valid_dataset, weights) + biases)
#         test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
#
#         num_steps = 801
#
#
#         def accuracy(predictions, labels):
#             return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))  # along axis=1 means by row.
#                     / predictions.shape[0])
#
#
#         with tf.Session(graph=graph) as session:
#             # This is a one-time operation which ensures the parameters get initialized as
#             # we described in the graph: random weights for the matrix, zeros for the
#             # biases.
#             tf.global_variables_initializer().run()
#             print('Initialized')
#             for step in range(num_steps):
#                 # Run the computations. We tell .run() that we want to run the optimizer,
#                 # and get the loss value and the training predictions returned as numpy
#                 # arrays.
#                 _, l, predictions = session.run([optimizer, loss, train_prediction])
#                 if (step % 100 == 0):
#                     print('Loss at step %d: %f' % (step, l))
#                     print('Training accuracy: %.1f%%' % accuracy(
#                         predictions, train_labels[:train_subset, :]))
#                     # Calling .eval() on valid_prediction is basically like calling run(), but
#                     # just to get that one numpy array. Note that it recomputes all its graph
#                     # dependencies.
#                     print('Validation accuracy: %.1f%%' % accuracy(
#                         valid_prediction.eval(), valid_labels))
#             print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
#
# ## SGD with LR// solver='sag' in sklearn.
# grid_lambda = np.array([1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001])
#
# batch_size = 128
# for regular_lambda in grid_lambda:
#     print('regularization coefficient:', regular_lambda)
#     graph = tf.Graph()
#     with graph.as_default():
#
#       # Input data. For the training data, we use a placeholder that will be fed
#       # at run time with a training minibatch.
#       tf_train_dataset = tf.placeholder(tf.float32,
#                                         shape=(batch_size, image_size * image_size))
#       tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
#       tf_valid_dataset = tf.constant(valid_dataset)
#       tf_test_dataset = tf.constant(test_dataset)
#
#       # Variables.
#       weights = tf.Variable(
#           tf.truncated_normal([image_size * image_size, num_labels]))
#       biases = tf.Variable(tf.zeros([num_labels]))
#
#       # Training computation.
#       logits = tf.matmul(tf_train_dataset, weights) + biases
#
#       # add regularization into loss
#       loss = tf.reduce_mean(
#           tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)
#           + regular_lambda*tf.nn.l2_loss(weights))
#
#       # Optimizer.
#       optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
#
#       # Predictions for the training, validation, and test data.
#       train_prediction = tf.nn.softmax(logits)
#       valid_prediction = tf.nn.softmax(
#           tf.matmul(tf_valid_dataset, weights) + biases)
#       test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
#
#       num_steps = 3001
#
#
#       def accuracy(predictions, labels):
#           return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))  # along axis=1 means by row.
#                   / predictions.shape[0])
#
#
#       with tf.Session(graph=graph) as session:
#           tf.global_variables_initializer().run()
#           print("Initialized")
#           for step in range(num_steps):
#               # Pick an offset within the training data, which has been randomized.
#               # Note: we could use better randomization across epochs.
#               offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
#               # Generate a minibatch.
#               batch_data = train_dataset[offset:(offset + batch_size), :]
#               batch_labels = train_labels[offset:(offset + batch_size), :]
#               # Prepare a dictionary telling the session where to feed the minibatch.
#               # The key of the dictionary is the placeholder node of the graph to be fed,
#               # and the value is the numpy array to feed to it.
#               feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
#               _, l, predictions = session.run(
#                   [optimizer, loss, train_prediction], feed_dict=feed_dict)
#               if (step % 500 == 0):
#                   print("Minibatch loss at step %d: %f" % (step, l))
#                   print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
#                   print("Validation accuracy: %.1f%%" % accuracy(
#                       valid_prediction.eval(), valid_labels))
#           print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


# best hyper-parameters [0.001, 0.001]
# Minibatch accuracy: 85.2%
# Validation accuracy: 87.1%
# Testing accuracy: 93.1%

# learning rate decay
# Minibatch accuracy: 83.6%
# Validation accuracy: 84.9%
# Testing accuracy: 91.3%

# dropout
# Minibatch accuracy: 73.4%
# Validation accuracy: 84.8%
# Testing accuracy: 91.3%
#
# grid_lambda = np.array([0.001])
# # batch_size = 50 # test for dropout
# batch_size = 128
# h1_length = 1024
# num_steps = 3001
# for regular_lambda1 in grid_lambda:
#     for regular_lambda2 in grid_lambda:
#         print('\nregularization coefficient:(%f, %f)' % (regular_lambda1, regular_lambda2))
#         graph = tf.Graph()
#         with graph.as_default():
#             keep_prob = tf.placeholder(dtype=tf.float32, shape=())
#             tf_train_dataset = tf.placeholder(dtype=tf.float32, shape=(batch_size, image_size**2))
#             tf_train_labels = tf.placeholder(dtype=tf.float32, shape=(batch_size, num_labels))
#             tf_valid_dataset = tf.constant(valid_dataset)
#             tf_test_dataset = tf.constant(test_dataset)
#
#             global_step = tf.Variable(0)  # count the number of steps taken.
#             # learning_rate = tf.train.exponential_decay(0.5, global_step, decay_rate=0.96, decay_steps=num_steps/(np.log(0.1/0.5)/np.log(0.96)))
#             learning_rate = 0.5
#             weights1 = tf.Variable(tf.truncated_normal([image_size**2, h1_length]))
#             biases1 = tf.Variable(tf.zeros([h1_length]))
#
#             #dropout: the size of mini-batch cannot be too small.
#             # I think we could use sigmoid function to substitute the softmax in case of NaN in loss.
#             # logits1 = tf.nn.relu(tf.matmul(tf_train_dataset, weights1) + biases1)
#             logits1 = tf.nn.dropout(x=tf.nn.relu(tf.matmul(tf_train_dataset, weights1) + biases1), keep_prob=keep_prob)
#
#             weights2 = tf.Variable(tf.truncated_normal([h1_length, num_labels]))
#             biases2 = tf.Variable(tf.zeros([num_labels]))
#
#             # logits2 = tf.nn.relu(tf.matmul(logits1, weights2) + biases2)
#             logits2 = tf.matmul(logits1, weights2) + biases2
#             loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits2, tf_train_labels)
#                                   + regular_lambda1*tf.nn.l2_loss(weights1)
#                                   + regular_lambda2*tf.nn.l2_loss(weights2))
#
#             # optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
#             optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
#
#             train_prediction = tf.nn.softmax(logits2)
#             valid_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, weights1) + biases1)
#                                                                   , weights2) + biases2)
#             test_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1)
#                                                                   , weights2) + biases2)
#
#
#             def accuracy(predictions, labels):
#                 return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]
#
#
#
#             with tf.Session(graph=graph) as session:
#                 tf.global_variables_initializer().run()
#                 for step in range(num_steps):
#                     offset = batch_size*step % (train_labels.shape[0]-batch_size)
#                     batch_data = train_dataset[offset:(offset+batch_size),:]
#                     batch_labels = train_labels[offset:(offset+batch_size),:]
#                     feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, keep_prob: 1}
#                     # feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, keep_prob: 0.5}
#                     _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
#                     if step % 500 == 0:
#                         print("Minibatch loss at step %d: %f" % (step, l))
#                         print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
#                         print("Validation accuracy: %.1f%%" % accuracy(
#                             valid_prediction.eval(), valid_labels))
#                 print('Testing accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))


# deep learning
# best_hypers
# '95.29': (0.001, 0.001, 0.0003, 0.0003, 0.001, 0.0003)
grid_lambda = np.array([[0.001],
                        [0.001],
                        [0.0003],
                        [0.0003],
                        [0.001],
                        [0.0003]
                        ])
# batch_size = 50 # test for dropout
# batch_size = 128
# batch_size = 50
batch_size = 256
h1_length = 1024
h2_length = h1_length//3
h3_length = h2_length//3
h4_length = h3_length//3
h5_length = h4_length//3
num_steps = 3001
best_hyper = {}
for regular_lambda1 in grid_lambda[0]:
    for regular_lambda2 in grid_lambda[1]:
        for regular_lambda3 in grid_lambda[2]:
            for regular_lambda4 in grid_lambda[3]:
                for regular_lambda5 in grid_lambda[4]:
                    for regular_lambda6 in grid_lambda[5]:
                        print('\nregularization coefficient:(%f, %f, %f, %f, %f, %f)' % (regular_lambda1, regular_lambda2, regular_lambda3, regular_lambda4, regular_lambda5, regular_lambda6))
                        graph = tf.Graph()
                        with graph.as_default():
                            keep_prob = tf.placeholder(dtype=tf.float32, shape=())
                            tf_train_dataset = tf.placeholder(dtype=tf.float32, shape=(batch_size, image_size**2))
                            tf_train_labels = tf.placeholder(dtype=tf.float32, shape=(batch_size, num_labels))
                            tf_valid_dataset = tf.constant(valid_dataset)
                            tf_test_dataset = tf.constant(test_dataset)

                            global_step = tf.Variable(0)  # count the number of steps taken.
                            initial_learning_rate = 0.5
                            final_learning_rate = 0.01
                            decay_rate = 0.96
                            learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_rate=decay_rate, decay_steps=num_steps/(np.log(final_learning_rate/initial_learning_rate)/np.log(decay_rate)))
                            # learning_rate = 0.5
                            # weights1 = tf.Variable(tf.truncated_normal([image_size**2, h1_length]))
                            weights1 = tf.get_variable('weights1', [image_size**2, h1_length], initializer=tf.contrib.layers.xavier_initializer())

                            biases1 = tf.Variable(tf.zeros([h1_length]))
                            logits1 = tf.nn.dropout(x=tf.nn.relu(tf.matmul(tf_train_dataset, weights1) + biases1), keep_prob=keep_prob)

                            weights2 = tf.get_variable('weights2', [h1_length, h2_length], initializer=tf.contrib.layers.xavier_initializer())
                            biases2 = tf.Variable(tf.zeros([h2_length]))
                            logits2 = tf.nn.dropout(x=tf.nn.relu(tf.matmul(logits1, weights2) + biases2), keep_prob=keep_prob)

                            weights3 = tf.get_variable('weights3', [h2_length, h3_length], initializer=tf.contrib.layers.xavier_initializer())
                            biases3 = tf.Variable(tf.zeros([h3_length]))
                            logits3 = tf.nn.dropout(x=tf.nn.relu(tf.matmul(logits2, weights3) + biases3), keep_prob=keep_prob)

                            weights4 = tf.get_variable('weights4', [h3_length, h4_length], initializer=tf.contrib.layers.xavier_initializer())
                            biases4 = tf.Variable(tf.zeros([h4_length]))
                            logits4 = tf.nn.dropout(x=tf.nn.relu(tf.matmul(logits3, weights4) + biases4), keep_prob=keep_prob)

                            weights5 = tf.get_variable('weights5', [h4_length, h5_length], initializer=tf.contrib.layers.xavier_initializer())
                            biases5 = tf.Variable(tf.zeros([h5_length]))
                            logits5 = tf.nn.dropout(x=tf.nn.relu(tf.matmul(logits4, weights5) + biases5), keep_prob=keep_prob)

                            weights6 = tf.get_variable('weights6', [h5_length, num_labels], initializer=tf.contrib.layers.xavier_initializer())
                            biases6 = tf.Variable(tf.zeros([num_labels]))
                            logits6 = tf.matmul(logits5, weights6) + biases6
                            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits6, tf_train_labels)
                                                  + regular_lambda1*tf.nn.l2_loss(weights1)
                                                  + regular_lambda2*tf.nn.l2_loss(weights2)
                                                  + regular_lambda3*tf.nn.l2_loss(weights3)
                                                  + regular_lambda4*tf.nn.l2_loss(weights4)
                                                  + regular_lambda5 * tf.nn.l2_loss(weights5)
                                                  + regular_lambda6 * tf.nn.l2_loss(weights6))

                            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

                            train_prediction = tf.nn.softmax(logits6)
                            valid_prediction = tf.nn.softmax(
                                tf.matmul(
                                tf.nn.relu(tf.matmul(
                                tf.nn.relu(tf.matmul(
                                tf.nn.relu(tf.matmul(
                                tf.nn.relu(tf.matmul(
                                tf.nn.relu(tf.matmul(tf_valid_dataset, weights1) + biases1)
                                                        , weights2) + biases2)
                                                        , weights3) + biases3)
                                                        , weights4) + biases4)
                                                        , weights5) + biases5)
                                                        , weights6) + biases6
                            )
                            test_prediction = tf.nn.softmax(
                                tf.matmul(
                                tf.nn.relu(tf.matmul(
                                tf.nn.relu(tf.matmul(
                                tf.nn.relu(tf.matmul(
                                tf.nn.relu(tf.matmul(
                                tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1)
                                                        , weights2) + biases2)
                                                        , weights3) + biases3)
                                                        , weights4) + biases4)
                                                        , weights5) + biases5)
                                                        , weights6) + biases6
                            )


                            def accuracy(predictions, labels):
                                return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]



                            with tf.Session(graph=graph) as session:
                                tf.global_variables_initializer().run()
                                for step in range(num_steps):
                                    offset = batch_size*step % (train_labels.shape[0]-batch_size)
                                    batch_data = train_dataset[offset:(offset+batch_size),:]
                                    batch_labels = train_labels[offset:(offset+batch_size),:]
                                    feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, keep_prob: 1}
                                    # feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, keep_prob: 0.5}
                                    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
                                    if step % 500 == 0:
                                        print("Minibatch loss at step %d: %f" % (step, l))
                                        print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                                        print("Validation accuracy: %.1f%%" % accuracy(
                                            valid_prediction.eval(), valid_labels))
                                testing_accuracy = accuracy(test_prediction.eval(), test_labels)
                                print('Testing accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
                                best_hyper[str(testing_accuracy)] = (regular_lambda1, regular_lambda2, regular_lambda3, regular_lambda4, regular_lambda5, regular_lambda6)
print('\nresults:\n', best_hyper)