#-*- coding: utf-8 -*-

import tensorflow as tf
import os

class Model(tf.keras.Model):
    """
    Abstract class for all trainable models that encapsulate multiple sub-models.
    """

    def __init__(self, *args, name='model', **kwargs):
        """
        Initialize keras Model class and necessary variables.
        """
        # The parent class
        super().__init__(name=name)
        # Initialize dictionary of submodels
        self.submodels = {}
        return

    def build(self, graph=None, inter_op_cores=1, intra_op_threads=1):
        """
        Initialize session AFTER all graphs have been built.
        """
        # Initialize session and global variables
        config = tf.ConfigProto()
        config.intra_op_parallelism_threads = intra_op_threads
        config.inter_op_parallelism_threads = inter_op_cores
        self.sess = tf.Session(graph=graph, config=config)
        self.sess.run(tf.global_variables_initializer())
        
        # Create savers for submodel variables
        self.savers = {}
        for name, model in self.submodels.items():
            self.savers[name] = tf.train.Saver(var_list=model.trainable_variables)

        return

    def print_variables(self):
        """
        Print layer names. Must be called after self.build()
        """
        for name, model in self.submodels.items():
            print('Model:', name)
            for var in model.trainable_variables:
                print('   ', var.name)
        return

    def save(self, outdir='checkpoints', model=None):
        """
        Save model weights to file.
        """
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        if model is None:
            for name, saver in self.savers.items():
                saver.save(self.sess, os.path.join(outdir, '%s.ckpt' % name))
        else:
            self.savers[model].save(self.sess, os.path.join(outdir, '%s.ckpt' % model))

        return

    def load(self, indir='checkpoints', model=None):
        """
        Load model weights from file.
        """
        if model is None:
            for name, saver in self.savers.items():
                saver.restore(self.sess, os.path.join(indir, '%s.ckpt' % name))
                print('Restoring', name)
        else:
            self.savers[model].restore(self.sess, os.path.join(indir, '%s.ckpt' % model))

        return
            

class DenseNet(tf.keras.Model):
    """
    Generic feedforward neural network.
    """

    def __init__(self, layer_sizes, name='net'):
        """
        Initialize and create layers.
        """
        # Initialize parent class
        super().__init__(name=name)

        # Create and store layers
        self.net_layers = []
        for count, size in enumerate(layer_sizes):
            # Layer names by depth count
            name = 'dense_%d' % count
            self.net_layers.append(
                tf.keras.layers.Dense(
                    size,
                    activation=None,
                    kernel_initializer='glorot_normal',
                    name=name
                )
            )
        self.n_layers = len(self.net_layers)

        return

    def call(self, inputs, activation='tanh', training=False, activate_outputs=False):
        """
        Pass inputs through network and generate an output. All layers except the last
        layer will pass through an activation function.

        NOTE: Do not call this directly. Use instance __call__() functionality.
        """
        # Cache activation function
        actfun = getattr(tf, activation)

        # Pass through all layers, use activations in all but last layer
        out = inputs
        for cnt, layer in enumerate(self.net_layers):
            out = layer(out)
            if cnt != (self.n_layers - 1):
                out = actfun(out)

        # Pass outputs through activation function
        if activate_outputs:
            out = actfun(out)

        # Done
        return out

# end of file
