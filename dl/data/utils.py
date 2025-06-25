import numpy as np
from dl import Variable
import math


class BatchLoader:

    def __init__(self, dataset, batch_size, shuffle=True, collate_fn=None):

        if collate_fn is None:
            collate_fn = default_collate

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn
        self.num_batches = math.ceil(len(dataset) / batch_size)

    def __iter__(self):

        N = len(self.dataset)
        if self.shuffle:
            indices = np.random.permutation(N)
        else:
            indices = np.arange(N)

        for i in range(0, N, self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]
            yield self.collate_fn(batch)

    def __len__(self):
        return self.num_batches


def default_collate(batch):
    """The default collate function expects a list of Variables or of tuples of Variables
        as input. For each tuple index, it iterates through the whole list, finding the Variables
        at that index and stacking them.

    Args:
        batch (list): A list of Variables, or of tuples of Variables.

    Returns:
        Union[Variable, tuple]: A Variable or a tuple of Variables.
    """
    if isinstance(batch[0], tuple):
        return tuple(
            Variable(np.stack([element.data for element in elements]))
            for elements in zip(*batch)
        )
    else:
        return Variable(np.stack([element.data for element in batch]))


class ComposeTransforms:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, X):

        for t in self.transforms:
            X = t(X)

        return X


def train_val_split(X, y, ratio=0.1, seed=42):
    np.random.seed(seed)

    N = X.data.shape[0]
    indices = np.random.permutation(N)
    split = int(N * (1 - ratio))
    train_idx = indices[:split]
    val_idx = indices[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    return X_train, y_train, X_val, y_val


def accuracy(features, labels):
    predictions = np.argmax(features.data, axis=1)
    targets = labels.data
    return np.sum(predictions == targets)
