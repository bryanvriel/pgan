#-*- coding: utf-8 -*-

import numpy as np

def atleast_2d(x):
    """
    Convenience function to ensure arrays are column vectors.
    """
    if x.ndim == 1:
        return x.reshape(-1, 1)
    elif x.ndim == 2:
        return x
    else:
        raise NotImplementedError('Input array has greater than 2 dimensions')


def train_test_indices(N, train_fraction=0.9, shuffle=True):
    """
    Convenience function to get train/test splits.
    """
    n_train = int(np.floor(train_fraction * N))
    if shuffle:
        ind = np.random.permutation(N)
    else:
        ind = np.arange(N, dtype=int)
    ind_train = ind[:n_train]
    ind_test = ind[n_train:]

    return ind_train, ind_test


class Data:
    """
    Class for representing and returning scattered points of solutions and coordinates.
    """

    def __init__(self, *args, train_fraction=0.9, batch_size=1024, shuffle=True, **kwargs):
        """
        Initialize dictionary of data and batching options. Data should be passed in
        via the kwargs dictionary.
        """
        # Check nothing has been passed in *args
        if len(args) > 0:
            raise ValueError('Data does not accept non-keyword arguments.')

        # Assume the coordinate T exists to generate train and test indices
        self.shuffle = shuffle
        self.n_data = kwargs['T'].size
        itrain, itest = train_test_indices(self.n_data,
                                           train_fraction=train_fraction,
                                           shuffle=shuffle)

        # Unpack the data for training
        self.keys = sorted(kwargs.keys())
        self._train = {}
        for key in self.keys:
            self._train[key] = kwargs[key][itrain]

        # Unpack the data for testing
        self._test = {}
        for key in self.keys:
            self._test[key] = kwargs[key][itest]

        # Cache training and batch size
        self.n_train = self._train['T'].size
        self.n_test = self.n_data - self.n_train
        self.batch_size = batch_size

        # Initialize counter for training data retrieval
        self._train_counter = 0

        return

    def train_batch(self):
        """
        Get a random batch of training data as a dictionary. Ensure that we cycle through
        complete set of training data (e.g., sample without replacement)
        """
        # If we've already reached the end of the training data, re-set counter with
        # optional re-shuffling of training data
        if self._train_counter >= self.n_train:
            self._train_counter = 0
            if self.shuffle:
                # Common shuffling indices
                ind = np.random.permutation(self.n_train)
                # Loop over data entries
                for key in self.keys:
                    self._train[key] = self._train[key][ind]

        # Construct slice for training data
        islice = slice(self._train_counter, self._train_counter + self.batch_size)
        result = {key: self._train[key][islice] for key in self.keys}

        # Update counter for training data
        self._train_counter += self.batch_size

        # All done
        return result

    def test_batch(self):
        """
        Get a random batch of testing data as a dictionary.
        """
        ind = np.random.choice(self.n_test, size=self.batch_size)
        return {key: self._test[key][ind] for key in self.keys}

    @property
    def train(self):
        """
        Get entire training set.
        """
        return self._train

    @train.setter
    def train(self, value):
        raise ValueError('Cannot set train variable.')

    @property
    def test(self):
        """
        Get entire testing set.
        """
        return self._test

    @test.setter
    def test(self, value):
        raise ValueError('Cannot set test variable.')
       
 
# end of file
