import os
import gzip
import functools
import operator
import numpy as np
import array
import struct
from urllib.parse import urljoin
from urllib.request import urlretrieve
from pathlib import Path

#DATASETS_URL = 'http://yann.lecun.com/exdb/mnist/'
DATASETS_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"

def one_hot_encode(labels:np.ndarray, num_classes:int):
    """One-hot encode the given labels.

    Parameters
    ----------
    labels : np.ndarray
        Array of shape (N, 1) containing the labels to one-hot encode

    num_classes : int
        Number of classes in the dataset

    Returns
    -------
    one_hot_labels : np.ndarray
        Array of shape (N, num_classes) containing the one-hot encoded labels
    """
    one_hot_labels = np.zeros((labels.shape[0], num_classes))
    one_hot_labels[np.arange(labels.shape[0]), labels.flatten()] = 1
    return one_hot_labels

def normalize(images:np.ndarray):
    """Normalize the given images to have zero mean and unit variance."""
    mean, std = np.mean(images), np.std(images)
    normalized_images = (images - mean) / std
    return normalized_images

def compute_accuracy(y_hat:np.ndarray, y:np.ndarray):
    """Compute the accuracy of the predictions.

    Parameters
    ----------
    y_hat : np.ndarray
        Array of shape (N, num_classes) containing the predicted probabilities for each class

    y : np.ndarray
        Array of shape (N, num_classes) containing the one-hot encoded true labels

    Returns
    -------
    accuracy : float
        Accuracy of the predictions, between 0 and 1
    """
    y_hat_labels = np.argmax(y_hat, axis=1)
    y_labels = np.argmax(y, axis=1)
    accuracy = np.mean(y_hat_labels == y_labels)
    return accuracy

class IdxDecodeError(ValueError):
    """Raised when an invalid idx file is parsed."""
    pass

class MNISTDataset():
    def __init__(
            self,
            split:str,
            dataset_directory:str = 'data/MNIST',
            batch_size:int = 1
    ):
        self.dataset_directory = Path(dataset_directory)
        self.dataset_directory.mkdir(parents=True, exist_ok=True)
    
        assert split in ['train', 'val'], "split must be either 'train' or 'val'"
        self.split = split
        self.batch_size = batch_size

        if self.split == 'train':
            self.images_fname = 'train-images-idx3-ubyte.gz'
            self.labels_fname = 'train-labels-idx1-ubyte.gz'
        else:
            self.images_fname = 't10k-images-idx3-ubyte.gz'
            self.labels_fname = 't10k-labels-idx1-ubyte.gz'

        self.images = self._download_and_parse_mnist_file(self.images_fname, target_dir=self.dataset_directory)
        self.labels = self._download_and_parse_mnist_file(self.labels_fname, target_dir=self.dataset_directory)

        self.reset()

    def _download_file(self, fname, target_dir, force=False):
        """Download fname from the datasets_url, and save it to target_dir,
        unless the file already exists, and force is False.

        Parameters
        ----------
        fname : str
            Name of the file to download

        target_dir : str
            Directory where to store the file

        force : bool
            Force downloading the file, if it already exists

        Returns
        -------
        fname : str
            Full path of the downloaded file
        """
        target_fname = os.path.join(target_dir, fname)

        if force or not os.path.isfile(target_fname):
            url = urljoin(DATASETS_URL, fname)
            urlretrieve(url, target_fname)

        return target_fname

    def _parse_idx(self, fd):
        """Parse an IDX file, and return it as a numpy array.

        Parameters
        ----------
        fd : file
            File descriptor of the IDX file to parse

        endian : str
            Byte order of the IDX file. See [1] for available options

        Returns
        -------
        data : numpy.ndarray
            Numpy array with the dimensions and the data in the IDX file

        1. https://docs.python.org/3/library/struct.html
            #byte-order-size-and-alignment
        """
        DATA_TYPES = {0x08: 'B',  # unsigned byte
                    0x09: 'b',  # signed byte
                    0x0b: 'h',  # short (2 bytes)
                    0x0c: 'i',  # int (4 bytes)
                    0x0d: 'f',  # float (4 bytes)
                    0x0e: 'd'}  # double (8 bytes)

        header = fd.read(4)
        if len(header) != 4:
            raise IdxDecodeError('Invalid IDX file, '
                                'file empty or does not contain a full header.')

        zeros, data_type, num_dimensions = struct.unpack('>HBB', header)

        if zeros != 0:
            raise IdxDecodeError('Invalid IDX file, '
                                'file must start with two zero bytes. '
                                'Found 0x%02x' % zeros)

        try:
            data_type = DATA_TYPES[data_type]
        except KeyError:
            raise IdxDecodeError('Unknown data type '
                                '0x%02x in IDX file' % data_type)

        dimension_sizes = struct.unpack('>' + 'I' * num_dimensions,
                                        fd.read(4 * num_dimensions))

        data = array.array(data_type, fd.read())
        data.byteswap()  # looks like array.array reads data as little endian

        expected_items = functools.reduce(operator.mul, dimension_sizes)
        if len(data) != expected_items:
            raise IdxDecodeError('IDX file has wrong number of items. '
                                'Expected: %d. Found: %d' % (expected_items,
                                                            len(data)))

        return np.array(data).reshape(dimension_sizes)

    def _download_and_parse_mnist_file(self, fname, target_dir=None, force=False):
        """Download the IDX file named fname from the URL specified in dataset_url
        and return it as a numpy array.

        Parameters
        ----------
        fname : str
            File name to download and parse

        target_dir : str
            Directory where to store the file

        force : bool
            Force downloading the file, if it already exists

        Returns
        -------
        data : numpy.ndarray
            Numpy array with the dimensions and the data in the IDX file
        """
        fname = self._download_file(fname, target_dir=target_dir, force=force)
        fopen = gzip.open if os.path.splitext(fname)[1] == '.gz' else open
        with fopen(fname, 'rb') as fd:
            return self._parse_idx(fd)
    
    def shuffle(self):
        self.sample_indices = np.arange(len(self.images))

    def reset(self):
        self.shuffle()
        np.random.shuffle(self.sample_indices)
        self.index = 0

    def __len__(self):
        return len(self.images) // self.batch_size

    def __iter__(self):
        return self  # Iterator is the object itself

    def __next__(self):
        if self.index >= len(self.images) // self.batch_size:
            raise StopIteration
        indices = self.sample_indices[self.index * self.batch_size:(self.index + 1) * self.batch_size]
        images, labels = self.images[indices].copy(), self.labels[indices].copy()

        images = np.expand_dims(images, 1) / 255.
        labels = np.expand_dims(labels, 1)

        self.index += 1
        return images, labels
    
if __name__ == "__main__":
    ds_train = MNISTDataset(split = 'train')
    pass