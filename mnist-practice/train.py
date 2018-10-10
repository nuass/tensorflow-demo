#coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

REGULARIZATION_RATE = 0.0001
LEARNING_RATE_DECAY = 0.99
flags = tf.app.flags

flags.DEFINE_integer('train_steps', 20000, 'Number of training steps to perform')
flags.DEFINE_integer('batch_size', 20, 'Training batch size ')
flags.DEFINE_string('data_dir', './mnist_data', 'Directory  for storing mnist data')
flags.DEFINE_string('checkpoint_dir', './model', 'Directory of model saving')
flags.DEFINE_string('model_name', 'mnist', 'Model saved name')
flags.DEFINE_string('summary_dir', './train_logs', 'Directory of logs saving ')
flags.DEFINE_float('learning_rate', '0.001','Number of learning rate to perform')
flags.DEFINE_string('start_checkpoint','','If specified, restore this pretrained model before any training.')
FLAGS = flags.FLAGS

def deepnn(x):
    #this is a LeNet-5 framwork
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])
    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)
    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)
    
    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        tf.add_to_collection('losses', regularizer(W_fc1))
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        tf.summary.histogram('activations', h_fc1)
    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
 

    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 10])
        #if regular==1.0:
        tf.add_to_collection('losses', regularizer(W_fc2))
        b_fc2 = bias_variable([10])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob

#Small values of L2 can help prevent overfitting the training data.
regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE,'regular')
#regular = tf.placeholder_with_default(1.0, shape=())

def conv2d(x, W):
    #conv2d returns a 2d convolution eayer with full stride.
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    #max_pool_2x2 downsamples a feature map by 2X
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                            strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
    #weight_variable generates a weight variable of a given shape.
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    #bias_variable generates a bias variable of a given shape.
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
   
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784])
        tf.summary.image('input_image', tf.reshape(x, [-1, 28, 28, 1]), 5)
        y_ = tf.placeholder(tf.float32, [None, 10])
        #tf.summary.scalar('input_label', y_,10)
    #global_step = tf.Variable(0, trainable=False)
    global_step = tf.train.get_or_create_global_step()
    increment_global_step = tf.assign(global_step, global_step + 1)
    with tf.name_scope('lr'):
        learning_rate = tf.train.exponential_decay(
                       FLAGS.learning_rate,
                       increment_global_step,
                       mnist.train.num_examples /FLAGS.batch_size,#decay step 
                       LEARNING_RATE_DECAY,
                       staircase=True)
        tf.summary.scalar('lr',learning_rate)
    y_conv, keep_prob = deepnn(x)
    # Define loss and optimizer
    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv)
        cross_entropy = tf.reduce_mean(cross_entropy)
        loss = cross_entropy + tf.add_n(tf.get_collection('losses'))
        tf.summary.scalar('loss', loss) 
    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
        tf.summary.scalar('accuracy', accuracy)
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.summary_dir,graph=tf.get_default_graph())
    saver = tf.train.Saver()
    start_step=1
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ###
        if FLAGS.start_checkpoint:
           saver = tf.train.Saver(tf.global_variables())
           saver.restore(sess,FLAGS.start_checkpoint)
           start_step = global_step.eval(session=sess)
        print('Training from step: %d ', start_step)
        ###
        for step in range(FLAGS.train_steps):
            batch = mnist.train.next_batch(FLAGS.batch_size)
            _,summary,steps = sess.run([train_step,merged,increment_global_step],feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            summary_writer.add_summary(summary, step)
            if step % 100 == 0:
               train_accuracy = accuracy.eval(feed_dict={ x: batch[0], y_: batch[1], keep_prob: 1.0})
               print('step %d, training accuracy %g' % (steps, train_accuracy))
               saver.save(sess, os.path.join(FLAGS.checkpoint_dir, FLAGS.model_name), global_step=steps)
    
        print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

if __name__ == '__main__':
   tf.reset_default_graph()
   tf.app.run()
