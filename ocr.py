import numpy as np
from scipy import ndimage
import pickle
import tensorflow as tf
from matplotlib import pyplot as plt
import sys


# 64 x 64, 95.64% at 98%
# architect: 8 convs + 3 FCL(LR, ReLU). maxout, channels: [48, 64, 128, 160], 192, 3072. dropout.
# conv: 5 x 5 zero padding, maxpooling 2 x 2, stride 2
# prediction: argmax_s(logP(S=s|H))
image_size = 64
num_labels_digit = 10
num_labels_length = 6
num_channels = 3
RECOGNITION_LENGTH = 5


def load_data(debug=True):
    if debug:
        subset_ratio = 1.0
        train_file = 'mini_train.pickle'
        test_file = 'mini_test.pickle'
    else:
        subset_ratio = 0.1
        train_file = 'train.pickle'
        test_file = 'test.pickle'

    with open(train_file, 'rb') as f:
        save = pickle.load(f)
        train_datasets = save['datasets'][0: int(save['datasets'].shape[0] * subset_ratio)]
        train_all_labels = save['labels'][0: int(save['labels'].shape[0] * subset_ratio)]
        del save  # hint to help gc free up memory

    with open(test_file, 'rb') as f:
        save = pickle.load(f)
        test_datasets = save['datasets'][0: int(save['datasets'].shape[0] * subset_ratio)]
        test_all_labels = save['labels'][0: int(save['labels'].shape[0] * subset_ratio)]
        del save  # hint to help gc free up memory
    return train_datasets, train_all_labels, test_datasets, test_all_labels

train_datasets, train_all_labels, test_datasets, test_all_labels = load_data(debug=False)

def reformat(datasets, labels, valid_ratio=0.0):
    datasets = datasets.reshape(
        (-1, image_size, image_size, num_channels)).astype(np.float32)
    labels = labels.reshape(-1, num_labels_length).astype(np.int32)
    valid_end = int(len(datasets) * valid_ratio)
    v_dataset = datasets[0: valid_end]
    v_labels = labels[0: valid_end]
    t_dataset = datasets[valid_end:]
    t_labels = labels[valid_end:]
    return v_dataset, v_labels, t_dataset, t_labels


valid_dataset, valid_labels,\
    train_dataset, train_labels = reformat(train_datasets, train_all_labels,
                                                                             valid_ratio=0.1)
_, _,\
    test_dataset, test_labels = reformat(test_datasets, test_all_labels)

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


def final_accuracy(length, digit_1, digit_2, digit_3, digit_4, digit_5, labels):
    digits = np.array([digit_1, digit_2, digit_3, digit_4, digit_5])
    num_success = 0
    for idx in range(length.shape[0]):
        max_L = 0
        predictions = []
        log_prob_digits = 0
        max_log_prob_seq = -sys.maxsize - 1
        for cnt in range(num_labels_length):
            if cnt == 0:
                log_prob_digits = 0
            else:
                max_pos = np.argmax(digits[cnt - 1, idx])
                predictions.append(max_pos)
                log_prob_digits += digits[cnt - 1, idx, max_pos]

            log_prob_seq = length[idx, cnt] + log_prob_digits
            if log_prob_seq > max_log_prob_seq:
                max_L = cnt
                max_log_prob_seq = log_prob_seq
        predictions = predictions[:max_L]
        true_L = labels[idx, 0]
        isSucceeded = True
        if true_L == max_L:
            if true_L != 0:
                for cnt in range(true_L):
                    if predictions[cnt] != labels[idx, cnt+1]:
                        isSucceeded = False
                        break
        else:
            isSucceeded = False
        if isSucceeded:
            num_success += 1
    return 100.0 * num_success / length.shape[0]


batch_size = 32
patch_size = 5
# c1_depth = 48
# c2_depth = 64
# c3_depth = 128
# c4_depth = 160
# f5_local_depth = 192
# f6_depth = 3072
# f7_depth = 3072
c1_depth = 16
c2_depth = 32
c3_depth = 64
c4_depth = 128
f5_local_depth = 192
f6_depth = 1024
f7_depth = 1024

pooling_stride = 2
pooling_kernel_size = 2
num_steps = 2001
# num_steps = 10
graph = tf.Graph()

with graph.as_default():
    # inputs
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels_length = tf.placeholder(tf.int32, shape=(batch_size))
    tf_train_labels_digit1 = tf.placeholder(tf.int32, shape=(batch_size))
    tf_train_labels_digit2 = tf.placeholder(tf.int32, shape=(batch_size))
    tf_train_labels_digit3 = tf.placeholder(tf.int32, shape=(batch_size))
    tf_train_labels_digit4 = tf.placeholder(tf.int32, shape=(batch_size))
    tf_train_labels_digit5 = tf.placeholder(tf.int32, shape=(batch_size))
    condition_0 = tf.placeholder(tf.int32, shape=[])
    condition_1 = tf.placeholder(tf.int32, shape=[])
    condition_2 = tf.placeholder(tf.int32, shape=[])
    condition_3 = tf.placeholder(tf.int32, shape=[])
    condition_4 = tf.placeholder(tf.int32, shape=[])
    condition_5 = tf.placeholder(tf.int32, shape=[])

    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    no_loss = tf.constant(0.0)

    # dropout
    keep_prob = tf.placeholder(tf.float32)

    # learning rate decay
    global_step = tf.Variable(0)  # count the number of steps taken.
    initial_learning_rate = 0.001
    final_learning_rate = 0.0001
    decay_rate = 0.96
    learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_rate=decay_rate,
                                               decay_steps=num_steps / (
                                               np.log(final_learning_rate / initial_learning_rate) / np.log(
                                                   decay_rate)))

    # variables
    c1_weights = tf.Variable(tf.truncated_normal(
        shape=[patch_size, patch_size, num_channels, c1_depth], stddev=0.1
    ))
    c1_biases = tf.Variable(tf.zeros([c1_depth]))
    c2_weights = tf.Variable(tf.truncated_normal(
        shape=[patch_size, patch_size, c1_depth, c2_depth], stddev=0.1
    ))
    c2_biases = tf.Variable(tf.zeros([c2_depth]))
    c3_weights = tf.Variable(tf.truncated_normal(
        shape=[patch_size, patch_size, c2_depth, c3_depth], stddev=0.1
    ))
    c3_biases = tf.Variable(tf.zeros([c3_depth]))
    c4_weights = tf.Variable(tf.truncated_normal(
        shape=[patch_size, patch_size, c3_depth, c4_depth], stddev=0.1
    ))
    c4_biases = tf.Variable(tf.zeros([c4_depth]))
    
    # locally hidden layer.
    f5_local_weights = tf.Variable(tf.truncated_normal(
        shape=[(image_size//(pooling_stride**4))**2 * c4_depth, f5_local_depth], stddev=0.1
    ))
    f5_local_biases = tf.Variable(tf.constant(1.0, shape=[f5_local_depth]))

    f6_weights_length = tf.Variable(tf.truncated_normal(
        shape=[f5_local_depth, f6_depth], stddev=0.1
    ))
    f6_biases_length = tf.Variable(tf.constant(1.0, shape=[f6_depth]))
    f6_weights_digit_1 = tf.Variable(tf.truncated_normal(
        shape=[f5_local_depth, f6_depth], stddev=0.1
    ))
    f6_biases_digit_1 = tf.Variable(tf.constant(1.0, shape=[f6_depth]))
    f6_weights_digit_2 = tf.Variable(tf.truncated_normal(
        shape=[f5_local_depth, f6_depth], stddev=0.1
    ))
    f6_biases_digit_2 = tf.Variable(tf.constant(1.0, shape=[f6_depth]))
    f6_weights_digit_3 = tf.Variable(tf.truncated_normal(
        shape=[f5_local_depth, f6_depth], stddev=0.1
    ))
    f6_biases_digit_3 = tf.Variable(tf.constant(1.0, shape=[f6_depth]))
    f6_weights_digit_4 = tf.Variable(tf.truncated_normal(
        shape=[f5_local_depth, f6_depth], stddev=0.1
    ))
    f6_biases_digit_4 = tf.Variable(tf.constant(1.0, shape=[f6_depth]))
    f6_weights_digit_5 = tf.Variable(tf.truncated_normal(
        shape=[f5_local_depth, f6_depth], stddev=0.1
    ))
    f6_biases_digit_5 = tf.Variable(tf.constant(1.0, shape=[f6_depth]))

    f7_weights_length = tf.Variable(tf.truncated_normal(
        shape=[f6_depth, f7_depth], stddev=0.1
    ))
    f7_biases_length = tf.Variable(tf.constant(1.0, shape=[f7_depth]))
    f7_weights_digit_1 = tf.Variable(tf.truncated_normal(
        shape=[f6_depth, f7_depth], stddev=0.1
    ))
    f7_biases_digit_1 = tf.Variable(tf.constant(1.0, shape=[f7_depth]))
    f7_weights_digit_2 = tf.Variable(tf.truncated_normal(
        shape=[f6_depth, f7_depth], stddev=0.1
    ))
    f7_biases_digit_2 = tf.Variable(tf.constant(1.0, shape=[f7_depth]))
    f7_weights_digit_3 = tf.Variable(tf.truncated_normal(
        shape=[f6_depth, f7_depth], stddev=0.1
    ))
    f7_biases_digit_3 = tf.Variable(tf.constant(1.0, shape=[f7_depth]))
    f7_weights_digit_4 = tf.Variable(tf.truncated_normal(
        shape=[f6_depth, f7_depth], stddev=0.1
    ))
    f7_biases_digit_4 = tf.Variable(tf.constant(1.0, shape=[f7_depth]))
    f7_weights_digit_5 = tf.Variable(tf.truncated_normal(
        shape=[f6_depth, f7_depth], stddev=0.1
    ))
    f7_biases_digit_5 = tf.Variable(tf.constant(1.0, shape=[f7_depth]))

    read_out_weights_length = tf.Variable(tf.truncated_normal(
        shape=[f7_depth, num_labels_length], stddev=0.1
    ))
    read_out_biases_length = tf.Variable(tf.constant(1.0, shape=[num_labels_length]))
    read_out_weights_digit_1 = tf.Variable(tf.truncated_normal(
        shape=[f7_depth, num_labels_digit], stddev=0.1
    ))
    read_out_biases_digit_1 = tf.Variable(tf.constant(1.0, shape=[num_labels_digit]))
    read_out_weights_digit_2 = tf.Variable(tf.truncated_normal(
        shape=[f7_depth, num_labels_digit], stddev=0.1
    ))
    read_out_biases_digit_2 = tf.Variable(tf.constant(1.0, shape=[num_labels_digit]))
    read_out_weights_digit_3 = tf.Variable(tf.truncated_normal(
        shape=[f7_depth, num_labels_digit], stddev=0.1
    ))
    read_out_biases_digit_3 = tf.Variable(tf.constant(1.0, shape=[num_labels_digit]))
    read_out_weights_digit_4 = tf.Variable(tf.truncated_normal(
        shape=[f7_depth, num_labels_digit], stddev=0.1
    ))
    read_out_biases_digit_4 = tf.Variable(tf.constant(1.0, shape=[num_labels_digit]))
    read_out_weights_digit_5 = tf.Variable(tf.truncated_normal(
        shape=[f7_depth, num_labels_digit], stddev=0.1
    ))
    read_out_biases_digit_5 = tf.Variable(tf.constant(1.0, shape=[num_labels_digit]))


    # pooling
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, pooling_kernel_size, pooling_kernel_size, 1],
                              strides=[1, pooling_stride, pooling_stride, 1], padding='SAME')

    # model
    def conv_model(data):
        conv = tf.nn.conv2d(data, c1_weights, [1,1,1,1], padding='SAME')
        hidden = tf.nn.dropout(x=tf.nn.relu(conv + c1_biases), keep_prob=keep_prob)
        pooling = max_pool_2x2(hidden)
        
        conv = tf.nn.conv2d(pooling, c2_weights, [1,1,1,1], padding='SAME')
        hidden = tf.nn.dropout(x=tf.nn.relu(conv + c2_biases), keep_prob=keep_prob)
        pooling = max_pool_2x2(hidden)
        
        conv = tf.nn.conv2d(pooling, c3_weights, [1,1,1,1], padding='SAME')
        hidden = tf.nn.dropout(x=tf.nn.relu(conv + c3_biases), keep_prob=keep_prob)
        pooling = max_pool_2x2(hidden)

        conv = tf.nn.conv2d(pooling, c4_weights, [1,1,1,1], padding='SAME')
        hidden = tf.nn.dropout(x=tf.nn.relu(conv + c4_biases), keep_prob=keep_prob)
        pooling = max_pool_2x2(hidden)

        shape = pooling.get_shape().as_list()
        reshape = tf.reshape(pooling, [shape[0], shape[1] * shape[2] * shape[3]])
        return tf.nn.dropout(x=tf.nn.relu(tf.matmul(reshape, f5_local_weights) + f5_local_biases), keep_prob=keep_prob)
            
    def classifier_model(data, f6_weights, f6_biases, f7_weights, f7_biases, read_out_weights, read_out_biases):
        hidden = tf.nn.dropout(x=tf.nn.relu(tf.matmul(data, f6_weights) + f6_biases), keep_prob=keep_prob)
        hidden = tf.nn.dropout(x=tf.nn.relu(tf.matmul(hidden, f7_weights) + f7_biases), keep_prob=keep_prob)
        return tf.matmul(hidden, read_out_weights) + read_out_biases

    # loss function
    conv_output = conv_model(tf_train_dataset)
    
    logits_length = classifier_model(
            conv_output, f6_weights_length, f6_biases_length, f7_weights_length, f7_biases_length, \
            read_out_weights_length, read_out_biases_length
        )
    loss_length = tf.cond(tf.equal(condition_2, 1), lambda: no_loss, lambda: tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=tf.boolean_mask(logits_length, tf.not_equal(tf_train_labels_length, -1)),
            labels=tf.boolean_mask(tf_train_labels_length, tf.not_equal(tf_train_labels_length, -1)),
        )
    ))
    logits_digit_1 = classifier_model(
            conv_output, f6_weights_digit_1, f6_biases_digit_1, f7_weights_digit_1, f7_biases_digit_1,\
            read_out_weights_digit_1, read_out_biases_digit_1
        )
    loss_digit_1 = tf.cond(tf.equal(condition_2, 1), lambda: no_loss, lambda: tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=tf.boolean_mask(logits_digit_1, tf.not_equal(tf_train_labels_digit1, -1)),
            labels=tf.boolean_mask(tf_train_labels_digit1, tf.not_equal(tf_train_labels_digit1, -1)),
        )
    ))
    logits_digit_2 = classifier_model(
            conv_output, f6_weights_digit_2, f6_biases_digit_2, f7_weights_digit_2, f7_biases_digit_2,\
            read_out_weights_digit_2, read_out_biases_digit_2
        )
    loss_digit_2 = tf.cond(tf.equal(condition_2, 1), lambda: no_loss, lambda: tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=tf.boolean_mask(logits_digit_2, tf.not_equal(tf_train_labels_digit2, -1)),
            labels=tf.boolean_mask(tf_train_labels_digit2, tf.not_equal(tf_train_labels_digit2, -1)),
        )
    ))
    logits_digit_3 = classifier_model(
            conv_output, f6_weights_digit_3, f6_biases_digit_3, f7_weights_digit_3, f7_biases_digit_3,\
            read_out_weights_digit_3, read_out_biases_digit_3
        )
    loss_digit_3 = tf.cond(tf.equal(condition_3, 1), lambda: no_loss, lambda: tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=tf.boolean_mask(logits_digit_3, tf.not_equal(tf_train_labels_digit3, -1)),
            labels=tf.boolean_mask(tf_train_labels_digit3, tf.not_equal(tf_train_labels_digit3, -1)),
        )
    ))
    logits_digit_4 = classifier_model(
            conv_output, f6_weights_digit_4, f6_biases_digit_4, f7_weights_digit_4, f7_biases_digit_4,\
            read_out_weights_digit_4, read_out_biases_digit_4
        )
    loss_digit_4 = tf.cond(tf.equal(condition_4, 1), lambda: no_loss, lambda: tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=tf.boolean_mask(logits_digit_4, tf.not_equal(tf_train_labels_digit4, -1)),
            labels=tf.boolean_mask(tf_train_labels_digit4, tf.not_equal(tf_train_labels_digit4, -1)),
        )
    ))
    logits_digit_5 = classifier_model(
            conv_output, f6_weights_digit_5, f6_biases_digit_5, f7_weights_digit_5, f7_biases_digit_5,\
            read_out_weights_digit_5, read_out_biases_digit_5
        )
    loss_digit_5 = tf.cond(tf.equal(condition_5, 1), lambda: no_loss, lambda: tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=tf.boolean_mask(logits_digit_5, tf.not_equal(tf_train_labels_digit5, -1)),
            labels=tf.boolean_mask(tf_train_labels_digit5, tf.not_equal(tf_train_labels_digit5, -1)),
        )
    ))
    loss = tf.reduce_mean(loss_length + loss_digit_1 + loss_digit_2 + loss_digit_3 + loss_digit_4 + loss_digit_5)
    
    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)
    
    # prediction
    log_train_prediction_length = tf.nn.log_softmax(logits_length)
    log_train_prediction_digit_1 = tf.nn.log_softmax(logits_digit_1)
    log_train_prediction_digit_2 = tf.nn.log_softmax(logits_digit_2)
    log_train_prediction_digit_3 = tf.nn.log_softmax(logits_digit_3)
    log_train_prediction_digit_4 = tf.nn.log_softmax(logits_digit_4)
    log_train_prediction_digit_5 = tf.nn.log_softmax(logits_digit_5)

    valid_prediction_length = tf.nn.log_softmax(classifier_model(
        conv_model(tf_valid_dataset), f6_weights_length, f6_biases_length, f7_weights_length, f7_biases_length, \
        read_out_weights_length, read_out_biases_length
    ))
    valid_prediction_digit_1 = tf.nn.log_softmax(classifier_model(
        conv_model(tf_valid_dataset), f6_weights_digit_1, f6_biases_digit_1, f7_weights_digit_1, f7_biases_digit_1, \
        read_out_weights_digit_1, read_out_biases_digit_1
    ))
    valid_prediction_digit_2 = tf.nn.log_softmax(classifier_model(
        conv_model(tf_valid_dataset), f6_weights_digit_2, f6_biases_digit_2, f7_weights_digit_2, f7_biases_digit_2, \
        read_out_weights_digit_2, read_out_biases_digit_2
    ))
    valid_prediction_digit_3 = tf.nn.log_softmax(classifier_model(
        conv_model(tf_valid_dataset), f6_weights_digit_3, f6_biases_digit_3, f7_weights_digit_3, f7_biases_digit_3, \
        read_out_weights_digit_3, read_out_biases_digit_3
    ))
    valid_prediction_digit_4 = tf.nn.log_softmax(classifier_model(
        conv_model(tf_valid_dataset), f6_weights_digit_4, f6_biases_digit_4, f7_weights_digit_4, f7_biases_digit_4, \
        read_out_weights_digit_4, read_out_biases_digit_4
    ))
    valid_prediction_digit_5 = tf.nn.log_softmax(classifier_model(
        conv_model(tf_valid_dataset), f6_weights_digit_5, f6_biases_digit_5, f7_weights_digit_5, f7_biases_digit_5, \
        read_out_weights_digit_5, read_out_biases_digit_5
    ))

    test_prediction_length = tf.nn.log_softmax(classifier_model(
        conv_model(tf_test_dataset), f6_weights_length, f6_biases_length, f7_weights_length, f7_biases_length, \
        read_out_weights_length, read_out_biases_length
    ))
    test_prediction_digit_1 = tf.nn.log_softmax(classifier_model(
        conv_model(tf_test_dataset), f6_weights_digit_1, f6_biases_digit_1, f7_weights_digit_1, f7_biases_digit_1, \
        read_out_weights_digit_1, read_out_biases_digit_1
    ))
    test_prediction_digit_2 = tf.nn.log_softmax(classifier_model(
        conv_model(tf_test_dataset), f6_weights_digit_2, f6_biases_digit_2, f7_weights_digit_2, f7_biases_digit_2, \
        read_out_weights_digit_2, read_out_biases_digit_2
    ))
    test_prediction_digit_3 = tf.nn.log_softmax(classifier_model(
        conv_model(tf_test_dataset), f6_weights_digit_3, f6_biases_digit_3, f7_weights_digit_3, f7_biases_digit_3, \
        read_out_weights_digit_3, read_out_biases_digit_3
    ))
    test_prediction_digit_4 = tf.nn.log_softmax(classifier_model(
        conv_model(tf_test_dataset), f6_weights_digit_4, f6_biases_digit_4, f7_weights_digit_4, f7_biases_digit_4, \
        read_out_weights_digit_4, read_out_biases_digit_4
    ))
    test_prediction_digit_5 = tf.nn.log_softmax(classifier_model(
        conv_model(tf_test_dataset), f6_weights_digit_5, f6_biases_digit_5, f7_weights_digit_5, f7_biases_digit_5, \
        read_out_weights_digit_5, read_out_biases_digit_5
    ))

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print('Initialized')
        for step in range(num_steps):
            # feed dict
            offset = (step * batch_size) % (train_dataset.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
            batch_labels_length = train_labels[offset:(offset + batch_size), 0]
            if np.sum(batch_labels_length == -1) == batch_size:
                condition0 = 1
            else:
                condition0 = 0
            batch_labels_digit_1 = train_labels[offset:(offset + batch_size), 1]
            if np.sum(batch_labels_digit_1 == -1) == batch_size:
                condition1 = 1
            else:
                condition1 = 0
            batch_labels_digit_2 = train_labels[offset:(offset + batch_size), 2]
            if np.sum(batch_labels_digit_2 == -1) == batch_size:
                condition2 = 1
            else:
                condition2 = 0
            batch_labels_digit_3 = train_labels[offset:(offset + batch_size), 3]
            if np.sum(batch_labels_digit_3 == -1) == batch_size:
                condition3 = 1
            else:
                condition3 = 0
            batch_labels_digit_4 = train_labels[offset:(offset + batch_size), 4]
            if np.sum(batch_labels_digit_4 == -1) == batch_size:
                condition4 = 1
            else:
                condition4 = 0
            batch_labels_digit_5 = train_labels[offset:(offset + batch_size), 5]
            if np.sum(batch_labels_digit_5 == -1) == batch_size:
                condition5 = 1
            else:
                condition5 = 0
            print('conditions: ', condition0, condition1, condition2, condition3, condition4, condition5)
            feed_dict = {tf_train_dataset: batch_data,
                         tf_train_labels_length: batch_labels_length,
                         tf_train_labels_digit1: batch_labels_digit_1,
                         tf_train_labels_digit2: batch_labels_digit_2,
                         tf_train_labels_digit3: batch_labels_digit_3,
                         tf_train_labels_digit4: batch_labels_digit_4,
                         tf_train_labels_digit5: batch_labels_digit_5,
                         keep_prob: 0.5,
                         condition_0: condition0,
                         condition_1: condition1,
                         condition_2: condition2,
                         condition_3: condition3,
                         condition_4: condition4,
                         condition_5: condition5
                         }
            # run session
            _, l, log_predictions_length, log_predictions_digit_1,\
                log_predictions_digit_2, log_predictions_digit_3,\
                log_predictions_digit_4, log_predictions_digit_5,\
                loss_L, loss_1, loss_2, loss_3, loss_4, loss_5\
                = session.run(
                        [optimizer, loss,
                         log_train_prediction_length,
                         log_train_prediction_digit_1,
                         log_train_prediction_digit_2,
                         log_train_prediction_digit_3,
                         log_train_prediction_digit_4,
                         log_train_prediction_digit_5,
                         loss_length,
                         loss_digit_1,
                         loss_digit_2,
                         loss_digit_3,
                         loss_digit_4,
                         loss_digit_5
                         ], feed_dict=feed_dict)
            print('loss_L %s. loss_1 %s, loss_2 %s, loss_3 %s, loss_4 %s, loss_5 %s'
                  % (loss_L, loss_1, loss_2, loss_3, loss_4, loss_5))
            # cross validation and report
            if step % 50 == 0:
                print('learning rate:', learning_rate.eval())
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch accuracy: %.1f%%' % final_accuracy(
                    log_predictions_length,
                    log_predictions_digit_1,
                    log_predictions_digit_2,
                    log_predictions_digit_3,
                    log_predictions_digit_4,
                    log_predictions_digit_5,
                    train_labels[offset:(offset + batch_size)]))
                print('Validation accuracy: %.1f%%' % final_accuracy(
                    *session.run([
                        valid_prediction_length,
                        valid_prediction_digit_1,
                        valid_prediction_digit_2,
                        valid_prediction_digit_3,
                        valid_prediction_digit_4,
                        valid_prediction_digit_5
                    ], feed_dict={keep_prob:1.0}), valid_labels))
        # Test accuracy.
        print('Test accuracy: %.1f%%' % final_accuracy(
            *session.run([
                test_prediction_length,
                test_prediction_digit_1,
                test_prediction_digit_2,
                test_prediction_digit_3,
                test_prediction_digit_4,
                test_prediction_digit_5
            ], feed_dict={keep_prob: 1.0}), test_labels))
