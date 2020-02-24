#-*- coding: utf-8 -*-

from collections import OrderedDict
import pgan.tensorflow as tf

class MultiVariable:
    """
    Class for representing multi-component input and output variables.
    """

    def __init__(self, dtype=tf.float32, **kwargs):
        """
        Initialize the key-value stores of {variable_name: tf.placeholder}. The **kwargs
        can specify either:

            A) {variable_name: variable_dimension}
            B) {variable_name: Tensor}

        Note: Since Python 3.6, keyword argument order is preserved (like an OrderedDict).
        """
        self.vars = OrderedDict()
        for varname, value in kwargs.items():

            # Create a placeholder
            if isinstance(value, int):
                self.vars[varname] = tf.placeholder(dtype, shape=[None, value], name=varname)

            # Or store Tensor values
            elif isinstance(value, tf.Tensor):
                self.vars[varname] = value

            # Otherwise, incompatible input
            else:
                raise ValueError('Unsupported variable value.')

    def concat(self, var_list=None):
        """
        Concatenates individual variables along the last dimension.
        """
        # List all variables
        if var_list is None:
            values = self.values()

        # Or specific variables
        else:
            values = [self.vars[name] for name in var_list]

        # Concatenate and return
        return tf.concat(values=values, axis=-1)

    def make_feed_dict(self, batch, feed_dict=None):
        """
        Fill placeholder values from batch dictionary.
        """
        # Initialize empty feed dict if not provided
        new = False
        if feed_dict is None:
            new = True
            feed_dict = {}

        # Fill it
        for varname, placeholder in self.vars.items():
            try:
                feed_dict[placeholder] = batch[varname]
            except KeyError:
                raise ValueError('Could not link variable %s to value in batch' % varname)

        # Done
        if new:
            return feed_dict

    def keys(self):
        """
        Alias for MultiVariable.names().
        """
        return self.names()

    def names(self):
        """
        Return the variable names.
        """
        return list(self.vars.keys())

    def values(self):
        """
        Return the variable values.
        """
        return list(self.vars.values())

    def items(self):
        """
        Return the variable items.
        """
        return self.vars.items()
    
    def sum(self):
        """
        Returns sum over all variables. Tensorflow will broacast dimensions when possible.
        """
        return sum([value for value in self.vars.values()])

    def __getitem__(self, key):
        """
        Return specific variable value.
        """
        return self.vars[key]

    def __setitem__(self, key, value):
        """
        Set specific variable value.
        """
        self.vars[key] = value


# end of file
