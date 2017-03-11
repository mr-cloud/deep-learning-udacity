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

def check_dataset(pickle_file):
    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        train_datasets = save['train_datasets']
        plt.figure()
        plt.imshow(train_datasets[1])
        plt.show()

check_dataset('train.pickle')
# check_dataset('notMNIST.pickle')

with open('train.pickle', 'rb') as f:
    save = pickle.load(f)
    train_datasets = save['train_datasets']
    train_labels = save['train_labels']
    del save  # hint to help gc free up memory
with open('test.pickle', 'rb') as f:
    save = pickle.load(f)
    test_datasets = save['test_datasets']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory

image_size = 64
num_labels_digit = 10
num_labels_length = 6
num_channels = 3
RECOGNITION_LENGTH = 5


def reformat_labels(datasets, labels, valid_ratio=0):
    length_labels = (np.arange(num_labels_length) == labels[:, 0, np.newaxis]).astype(np.float32)
    digit_labels = np.ndarray(shape=(RECOGNITION_LENGTH, len(labels), num_labels_digit), dtype=np.float32)
    for idx in range(RECOGNITION_LENGTH):
        digit_labels[idx] = (np.arange(num_labels_digit) == labels[:, (idx+1), np.newaxis]).astype(np.float32)
    valid_end = int(len(datasets) * valid_ratio)
    valid_dataset = datasets[0: valid_end]
    valid_length_labels = length_labels[0: valid_end]
    valid_digit_labels = digit_labels[:, 0: valid_end, :]
    train_dataset = datasets[valid_end:]
    train_length_labels = length_labels[valid_end:]
    train_digit_labels = digit_labels[:, valid_end:, :]
    return valid_dataset, valid_length_labels, valid_digit_labels, \
           train_dataset, train_length_labels, train_digit_labels


valid_dataset, valid_length_labels, valid_digit_labels,\
    train_dataset, train_length_labels, train_digit_labels = reformat_labels(train_datasets, train_labels, valid_ratio=0.1)
_, _, _,\
    test_dataset, test_length_labels, test_digit_labels = reformat_labels(train_datasets, train_labels)

print('Training set', train_dataset.shape, train_length_labels.shape, train_digit_labels.shape)
print('Validation set', valid_dataset.shape, valid_length_labels.shape, valid_digit_labels.shape)
print('Test set', test_dataset.shape, test_length_labels.shape, test_digit_labels.shape)


def accuracy(predictions, labels):
    predictions = softmax(predictions)
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


batch_size = 32
patch_size = 5
c1_depth = 48
c2_depth = 64
c3_depth = 128
c4_depth = 160
f5_local_depth = 192
f6_depth = 3072
f7_depth = 3072
pooling_stride = 2
pooling_kernel_size = 2

# num_steps = 2001
num_steps = 10

graph = tf.Graph()


def log_softmax(x):
    return x - np.log(np.sum(np.exp(x), axis=1, keepdims=True))


def final_accuracy(length, digit_1, digit_2, digit_3, digit_4, digit_5, length_labels, digit_labels):
    # logsoftmax(logits)
    length = log_softmax(length)
    digit_1 = log_softmax(digit_1)
    digit_2 = log_softmax(digit_2)
    digit_3 = log_softmax(digit_3)
    digit_4 = log_softmax(digit_4)
    digit_5 = log_softmax(digit_5)
    digits = np.array([digit_1, digit_2, digit_3, digit_4, digit_5])
    num_success = 0
    for idx in range(length.shape[0]):
        max_L = 0
        predictions = []
        log_prob_digits = 0
        log_prob_seq = 0
        max_log_prob_seq = -sys.maxsize - 1
        for cnt in range(num_labels_length):
            if cnt == 0:
                log_prob_digits = 0
            else:
                max_pos = np.argmax(digits[cnt - 1, idx], 1)
                predictions.append(max_pos)
                log_prob_digits += digits[cnt - 1, idx, max_pos]

            log_prob_seq = length[idx, cnt] + log_prob_digits
            if log_prob_seq > max_log_prob_seq:
                max_L = cnt
                max_log_prob_seq = log_prob_seq
        predictions = predictions[:max_L]
        true_L = np.argmax(length_labels[idx])
        isSucceeded = True
        if true_L == max_L:
            if true_L != 0:
                for cnt in range(true_L):
                    if predictions[cnt] != np.argmax(digit_labels[cnt, idx]):
                        isSucceeded = False
                        break
        else:
            isSucceeded = False
        if isSucceeded:
            num_success += 1
    return 100.0 * num_success / length.shape[0]



def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)


with graph.as_default():
    # inputs
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size, image_size))
    tf_train_labels_length = tf.placeholder(tf.float32, shape=(batch_size, num_labels_length))
    tf_train_labels_digit1 = tf.placeholder(tf.float32, shape=(batch_size, num_labels_digit))
    tf_train_labels_digit2 = tf.palceholder(tf.float32, shape=(batch_size, num_labels_digit))
    tf_train_labels_digit3 = tf.palceholder(tf.float32, shape=(batch_size, num_labels_digit))
    tf_train_labels_digit4 = tf.palceholder(tf.float32, shape=(batch_size, num_labels_digit))
    tf_train_labels_digit5 = tf.palceholder(tf.float32, shape=(batch_size, num_labels_digit))

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
        shape=[(image_size//(pooling_stride**4)**2) * c4_depth, f5_local_depth], stddev=0.1
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
    loss_length = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits_length, labels=tf_train_labels_length)
    )
    logits_digit_1 = classifier_model(
            conv_output, f6_weights_digit_1, f6_biases_digit_1, f7_weights_digit_1, f7_biases_digit_1,\
            read_out_weights_digit_1, read_out_biases_digit_1
        )
    loss_digit_1 = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits_digit_1, labels=tf_train_labels_digit1)
    )
    logits_digit_2 = classifier_model(
            conv_output, f6_weights_digit_2, f6_biases_digit_2, f7_weights_digit_2, f7_biases_digit_2,\
            read_out_weights_digit_2, read_out_biases_digit_2
        )
    loss_digit_2 = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits_digit_2, labels=tf_train_labels_digit2)
    )
    logits_digit_3 = classifier_model(
            conv_output, f6_weights_digit_3, f6_biases_digit_3, f7_weights_digit_3, f7_biases_digit_3,\
            read_out_weights_digit_3, read_out_biases_digit_3
        )
    loss_digit_3 = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits_digit_3, labels=tf_train_labels_digit3)
    )
    logits_digit_4 = classifier_model(
            conv_output, f6_weights_digit_4, f6_biases_digit_4, f7_weights_digit_4, f7_biases_digit_4,\
            read_out_weights_digit_4, read_out_biases_digit_4
        )
    loss_digit_4 = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits_digit_4, labels=tf_train_labels_digit4)
    )
    logits_digit_5 = classifier_model(
            conv_output, f6_weights_digit_5, f6_biases_digit_5, f7_weights_digit_5, f7_biases_digit_5,\
            read_out_weights_digit_5, read_out_biases_digit_5
        )
    loss_digit_5 = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits_digit_5, labels=tf_train_labels_digit5)
    )
    loss = tf.reduce_mean(loss_length + loss_digit_1 + loss_digit_2 + loss_digit_3 + loss_digit_4 + loss_digit_5)
    
    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)
    
    # prediction
    train_prediction_length = logits_length
    train_prediction_digit_1 = logits_digit_1
    train_prediction_digit_2 = logits_digit_2
    train_prediction_digit_3 = logits_digit_3
    train_prediction_digit_4 = logits_digit_4
    train_prediction_digit_5 = logits_digit_5
    
    valid_prediction_length = classifier_model(
        conv_model(tf_valid_dataset), f6_weights_length, f6_biases_length, f7_weights_length, f7_biases_length, \
        read_out_weights_length, read_out_biases_length
    )    
    valid_prediction_digit_1 = classifier_model(
        conv_model(tf_valid_dataset), f6_weights_digit_1, f6_biases_digit_1, f7_weights_digit_1, f7_biases_digit_1, \
        read_out_weights_digit_1, read_out_biases_digit_1
    )
    valid_prediction_digit_2 = classifier_model(
        conv_model(tf_valid_dataset), f6_weights_digit_2, f6_biases_digit_2, f7_weights_digit_2, f7_biases_digit_2, \
        read_out_weights_digit_2, read_out_biases_digit_2
    )
    valid_prediction_digit_3 = classifier_model(
        conv_model(tf_valid_dataset), f6_weights_digit_3, f6_biases_digit_3, f7_weights_digit_3, f7_biases_digit_3, \
        read_out_weights_digit_3, read_out_biases_digit_3
    )
    valid_prediction_digit_4 = classifier_model(
        conv_model(tf_valid_dataset), f6_weights_digit_4, f6_biases_digit_4, f7_weights_digit_4, f7_biases_digit_4, \
        read_out_weights_digit_4, read_out_biases_digit_4
    )
    valid_prediction_digit_5 = classifier_model(
        conv_model(tf_valid_dataset), f6_weights_digit_5, f6_biases_digit_5, f7_weights_digit_5, f7_biases_digit_5, \
        read_out_weights_digit_5, read_out_biases_digit_5
    )

    test_prediction_length = classifier_model(
        conv_model(tf_test_dataset), f6_weights_length, f6_biases_length, f7_weights_length, f7_biases_length, \
        read_out_weights_length, read_out_biases_length
    )
    test_prediction_digit_1 = classifier_model(
        conv_model(tf_test_dataset), f6_weights_digit_1, f6_biases_digit_1, f7_weights_digit_1, f7_biases_digit_1, \
        read_out_weights_digit_1, read_out_biases_digit_1
    )
    test_prediction_digit_2 = classifier_model(
        conv_model(tf_test_dataset), f6_weights_digit_2, f6_biases_digit_2, f7_weights_digit_2, f7_biases_digit_2, \
        read_out_weights_digit_2, read_out_biases_digit_2
    )
    test_prediction_digit_3 = classifier_model(
        conv_model(tf_test_dataset), f6_weights_digit_3, f6_biases_digit_3, f7_weights_digit_3, f7_biases_digit_3, \
        read_out_weights_digit_3, read_out_biases_digit_3
    )
    test_prediction_digit_4 = classifier_model(
        conv_model(tf_test_dataset), f6_weights_digit_4, f6_biases_digit_4, f7_weights_digit_4, f7_biases_digit_4, \
        read_out_weights_digit_4, read_out_biases_digit_4
    )
    test_prediction_digit_5 = classifier_model(
        conv_model(tf_test_dataset), f6_weights_digit_5, f6_biases_digit_5, f7_weights_digit_5, f7_biases_digit_5, \
        read_out_weights_digit_5, read_out_biases_digit_5
    )

    
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print('Initialized')
    for step in range(num_steps):
        # feed dict
        offset = (step * batch_size) % (train_dataset.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels_length = train_length_labels[offset:(offset + batch_size), :]
        batch_labels_digit_1 = train_digit_labels[0, offset:(offset + batch_size), :]
        batch_labels_digit_2 = train_digit_labels[1, offset:(offset + batch_size), :]
        batch_labels_digit_3 = train_digit_labels[2, offset:(offset + batch_size), :]
        batch_labels_digit_4 = train_digit_labels[3, offset:(offset + batch_size), :]
        batch_labels_digit_5 = train_digit_labels[4, offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset: batch_data,
                     tf_train_labels_length: batch_labels_length,
                     tf_train_labels_digit1: batch_labels_digit_1,
                     tf_train_labels_digit2: batch_labels_digit_2,
                     tf_train_labels_digit3: batch_labels_digit_3,
                     tf_train_labels_digit4: batch_labels_digit_4,
                     tf_train_labels_digit5: batch_labels_digit_5,
                     keep_prob: 0.5
                     }
        # run session
        _, l, predictions_length, \
            predictions_digit_1, predictions_digit_2, predictions_digit_3, predictions_digit_4, predictions_digit_5 \
            = session.run(
                [optimizer, loss,
                 train_prediction_length,
                 train_prediction_digit_1,
                 train_prediction_digit_2,
                 train_prediction_digit_3,
                 train_prediction_digit_4,
                 train_prediction_digit_5,
                 ], feed_dict=feed_dict)
        # cross validation and report
        if (step % 50 == 0):
            print('learning rate:', learning_rate.eval())
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy for length: %.1f%%' % accuracy(predictions_length, batch_labels_length))
            print('Minibatch accuracy for digit_1: %.1f%%' % accuracy(predictions_digit_1, batch_labels_digit_1))
            print('Minibatch accuracy for digit_2: %.1f%%' % accuracy(predictions_digit_2, batch_labels_digit_2))
            print('Minibatch accuracy for digit_3: %.1f%%' % accuracy(predictions_digit_3, batch_labels_digit_3))
            print('Minibatch accuracy for digit_4: %.1f%%' % accuracy(predictions_digit_4, batch_labels_digit_4))
            print('Minibatch accuracy for digit_5: %.1f%%' % accuracy(predictions_digit_5, batch_labels_digit_5))
            print('Minibatch accuracy: %.1f%%' % final_accuracy(
                predictions_length,
                predictions_digit_1,
                predictions_digit_2,
                predictions_digit_3,
                predictions_digit_4,
                predictions_digit_5,
                train_length_labels[offset:(offset + batch_size), :],
                train_digit_labels[:, offset:(offset + batch_size), :]))

            print('Validation accuracy: %.1f%%' % final_accuracy(
                valid_prediction_length.eval(), 
                valid_prediction_digit_1.eval(),
                valid_prediction_digit_2.eval(),
                valid_prediction_digit_3.eval(),
                valid_prediction_digit_4.eval(),
                valid_prediction_digit_5.eval(),
                valid_length_labels, valid_digit_labels))
    # F-score.
    print('Test accuracy: %.1f%%' % final_accuracy(
        test_prediction_length.eval(),
        test_prediction_digit_1.eval(),
        test_prediction_digit_2.eval(),
        test_prediction_digit_3.eval(),
        test_prediction_digit_4.eval(),
        test_prediction_digit_5.eval(),
        test_length_labels, test_digit_labels))
