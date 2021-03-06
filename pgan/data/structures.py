#-*- coding: utf-8 -*-

import numpy as np
import h5py
import os

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


def train_test_indices(N, train_fraction=0.9, shuffle=True, rng=None):
    """
    Convenience function to get train/test splits.
    """
    n_train = int(np.floor(train_fraction * N))
    if shuffle:
        assert rng is not None, 'Must pass in a random number generator'
        ind = rng.permutation(N)
    else:
        ind = np.arange(N, dtype=int)
    ind_train = ind[:n_train]
    ind_test = ind[n_train:]

    return ind_train, ind_test


class Data:
    """
    Class for representing and returning scattered points of solutions and coordinates.
    """

    def __init__(self, *args, train_fraction=0.9, batch_size=1024, shuffle=True,
                 seed=None, split_seed=None, **kwargs):
        """
        Initialize dictionary of data and batching options. Data should be passed in
        via the kwargs dictionary.
        """
        # Check nothing has been passed in *args
        if len(args) > 0:
            raise ValueError('Data does not accept non-keyword arguments.')

        # Create a random number generator
        self.rng = np.random.RandomState(seed=seed)
        if split_seed is not None:
            self.split_rng = np.random.RandomState(seed=split_seed)
        else:
            self.split_rng = np.random.RandomState(seed=seed)

        # Assume the coordinate T exists to determine data size
        self.shuffle = shuffle
        self.n_data = kwargs['T'].shape[0]
    
        # Generate train/test indices
        itrain, itest = train_test_indices(self.n_data,
                                           train_fraction=train_fraction,
                                           shuffle=shuffle,
                                           rng=self.split_rng)

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
        self.n_train = self._train['T'].shape[0]
        self.n_test = self.n_data - self.n_train
        self.batch_size = batch_size
        self.n_batches = int(np.ceil(self.n_train / self.batch_size))

        # Initialize training indices (data have already been shuffle, so only need arange here)
        self._itrain = np.arange(self.n_train, dtype=int)

        # Initialize counter for training data retrieval
        self._train_counter = 0

        return

    def train_batch(self):
        """
        Get a random batch of training data as a dictionary. Ensure that we cycle through
        complete set of training data (e.g., sample without replacement)
        """
        # If we've already reached the end of the training data, re-set counter with
        # optional re-shuffling of training indices
        if self._train_counter >= self.n_train:
            self._train_counter = 0
            if self.shuffle:
                self._itrain = self.rng.permutation(self.n_train)

        # Construct slice for training data indices
        islice = slice(self._train_counter, self._train_counter + self.batch_size)
        indices = self._itrain[islice]

        # Get training data
        result = {key: self._train[key][indices] for key in self.keys}

        # Update counter for training data
        self._train_counter += self.batch_size

        # All done
        return result

    def test_batch(self, batch_size=None):
        """
        Get a random batch of testing data as a dictionary.
        """
        batch_size = batch_size or self.batch_size
        ind = self.rng.choice(self.n_test, size=batch_size)
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


class H5Data:
    """
    Class for representing and returning scattered points of solutions and coordinates
    stored in an HDF5 file.
    """

    def __init__(self, h5file, keys, root='/', train_fraction=0.9, batch_size=1024,
                 shuffle=True, seed=None, **kwargs):
        """
        Initialize dictionary of data and batching options. Data should be passed in
        via the kwargs dictionary.
        """
        # Open HDF5 file
        if not os.path.isfile(h5file):
            raise FileNotFoundError('Cannot open HDF5 file %s' % h5file)
        self.fid = h5py.File(h5file, 'r')

        # Cache keys for datasets we wish to analyze
        self.keys = keys

        # Cache the root HDF5 path containing our datasets
        self.root = root

        # Create a random number generator
        self.rng = np.random.RandomState(seed=seed)

        # Assume the dataset T exists to determine data size
        self.shuffle = shuffle
        self.n_data = self.fid[os.path.join(root, 'T')].shape[0]
    
        # Generate train/test indices
        itrain, itest = train_test_indices(self.n_data,
                                           train_fraction=train_fraction,
                                           shuffle=shuffle,
                                           rng=self.rng)

        # Cache training and batch size
        self.n_train = len(itrain)
        self.n_test = len(itest)
        self.batch_size = batch_size
        self.n_batches = int(np.ceil(self.n_train / self.batch_size))

        # Save indices
        self._itrain = itrain
        self._itest = itest

        # Initialize counter for training data retrieval
        self._train_counter = 0

        return

    def __del__(self):
        self.fid.close()

    def train_batch(self):
        """
        Get a random batch of training data as a dictionary. Ensure that we cycle through
        complete set of training data (e.g., sample without replacement)
        """
        # If we've already reached the end of the training data, re-set counter with
        # optional re-shuffling of training indices
        if self._train_counter >= self.n_train:
            self._train_counter = 0
            if self.shuffle:
                self._itrain = self.rng.permutation(self._itrain)

        # Construct slice for training data indices
        islice = slice(self._train_counter, self._train_counter + self.batch_size)

        # Sort sliced training indices (need sorting due to h5py limitations)
        indices = np.sort(self._itrain[islice])

        # Get training data
        result = {key: self.fid[os.path.join(self.root, key)][indices,...] for key in self.keys}

        # Update counter for training data
        self._train_counter += self.batch_size

        # All done
        return result

    def test_batch(self):
        """
        Get a random batch of testing data as a dictionary.
        """
        # Make random test indices
        indices = np.sort(self.rng.choice(self._itest, size=self.batch_size, replace=False))

        # Get test data
        result = {key: self.fid[os.path.join(self.root, key)][indices,...] for key in self.keys}

        # All done
        return result
    
    @property
    def test(self):
        """
        Get entire testing set.
        """
        ind = np.sort(self.itest)
        return {key: self.fid[os.path.join(self.root, key)][ind] for key in self.keys}

    @test.setter
    def test(self, value):
        raise ValueError('Cannot set test variable.')

 
# end of file
