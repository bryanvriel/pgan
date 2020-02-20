#-*- coding: utf-8 -*-

import pgan.tensorflow as tf
from collections import OrderedDict
import shutil
import os

class Summary:
    """
    Convenience class for writing tensorflow summaries to be displayed with Tensorboard.
    Writes train and test loss values to the same plot.
    """

    def __init__(self, sess, loss_dict, outdir='summaries'):

        # Clean or ensure output directory exists
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        else:
            shutil.rmtree(outdir)
            os.makedirs(outdir)

        # Save copy of loss dictionary
        self.loss_dict = loss_dict
        self.loss_names = list(loss_dict.keys())

        # Initialize empty (ordered) dictionaries
        self.placeholders = OrderedDict()
        self.train_writers = OrderedDict()
        self.test_writers = OrderedDict()
        self.summaries = OrderedDict()

        # Iterate over loss items in loss dictionary
        summaries = []
        for loss_name, loss_node in loss_dict.items():
            
            # Create placeholder for loss value
            ph_name = '%s_value' % loss_name
            self.placeholders[loss_name] = tf.placeholder(tf.float32, name=ph_name)

            # Create tensorflow summary from loss value
            self.summaries[loss_name] = tf.summary.scalar(loss_name, self.placeholders[loss_name])

            # Create writers for train and test summaries
            loss_outdir = os.path.join(outdir, loss_name)
            self.train_writers[loss_name] = tf.summary.FileWriter(
                os.path.join(loss_outdir, 'train'), sess.graph
            )
            self.test_writers[loss_name] = tf.summary.FileWriter(
                os.path.join(loss_outdir, 'test'), sess.graph
            )

        # Merge all summaries into a single summary object
        #self.summary = tf.summary.merge(summaries)

    def write_summary(self, sess, feed_dict, iternum, stype='train'):

        # Evaluate loss nodes with feed dict
        loss_values = sess.run(list(self.loss_dict.values()), feed_dict=feed_dict)

        # Construct feed dictionary for loss placeholders
        feed_dict = {}
        for cnt, loss_name in enumerate(self.loss_names):
            feed_dict[self.placeholders[loss_name]] = loss_values[cnt]

        # Evaluate summary
        #summ = sess.run(self.summary, feed_dict=feed_dict)

        # Point to right summary writers
        if stype == 'train':
            writers = self.train_writers
        elif stype == 'test':
            writers = self.test_writers
        else:
            raise ValueError('Invalid stype for specifying summary writer.')

        # Add summaries and flush
        for loss_name in self.loss_names:
            summ = sess.run(self.summaries[loss_name], feed_dict)
            writers[loss_name].add_summary(summ, iternum)
            writers[loss_name].flush()

        return loss_values


# end of file
