import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from train import deepnn

MODEL_SAVE_PATH='./model'
TEST_LOGS_PATH='./test_logs'
def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, 784])
        sample_images = tf.summary.image('input_image', tf.reshape(x, [-1, 28, 28, 1]), 10)
        y_ = tf.placeholder(tf.float32, [None, 10])
        tf.summary.text('label',tf.as_string(tf.argmax(mnist.test.labels[:10],axis=0)))

        y,keep_prob = deepnn(x)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        tf.summary.text('prediction',tf.as_string(tf.argmax(y,1)[:10]))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        saver = tf.train.Saver()
        merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(TEST_LOGS_PATH,graph=tf.get_default_graph())
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt and ckpt.all_model_checkpoint_paths:
               for model_step in ckpt.all_model_checkpoint_paths:
                   saver.restore(sess, model_step)
                   accuracy_score,summary = sess.run([accuracy,merged],
                           feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
                   print("Restore model:%s , test accuracy = %g" % (model_step, accuracy_score))
        
               summary_writer.add_summary(summary)
            else:
                print('No checkpoint file found')
                return

def main(argv=None):
    mnist = input_data.read_data_sets("./mnist_data", one_hot=True)
    evaluate(mnist)
  
if __name__ == '__main__':
    
    tf.app.run()
