#-*- coding: utf-8 -*-

import pgan.tensorflow as tf
import shutil
import os

class Summary:
    """
    Convenience class for writing tensorflow summaries to be displayed with Tensorboard.
    Writes train and test loss values to the same plot.
    """

    def __init__(self, sess, loss_name='loss', outdir='summaries'):

        # Create placeholder for loss value
        ph_name = '%s_value' % loss_name
        self.loss_ph = tf.placeholder(tf.float32, name=ph_name)

        # Create tensorflow summary from loss value
        self.summary = tf.summary.scalar(loss_name, self.loss_ph)

        # Clean or ensure output directory exists
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        else:
            shutil.rmtree(outdir)
            os.makedirs(outdir)

        # Create writers for train and test summaries
        self.train_writer = tf.summary.FileWriter(os.path.join(outdir, 'train'), sess.graph)
        self.test_writer = tf.summary.FileWriter(os.path.join(outdir, 'test'), sess.graph)

    def write_train_summary(self, sess, loss_node, feed_dict, iternum):

        # Evaluate loss node with feed dict
        loss_value = sess.run(loss_node, feed_dict=feed_dict)

        # Evaluate summary
        summ = sess.run(self.summary, feed_dict={self.loss_ph: loss_value})
        self.train_writer.add_summary(summ, iternum)
        self.train_writer.flush()

        return loss_value

    def write_test_summary(self, sess, loss_node, feed_dict, iternum):

        # Evaluate loss node with feed dict
        loss_value = sess.run(loss_node, feed_dict=feed_dict)

        # Evaluate summary
        summ = sess.run(self.summary, feed_dict={self.loss_ph: loss_value})
        self.test_writer.add_summary(summ, iternum)
        self.test_writer.flush()

        return loss_value


# end of file
