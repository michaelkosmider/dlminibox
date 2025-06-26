import numpy as np
from dl import Variable


class ToFloat:

    def __call__(self, X):
        return X.astype(np.float32) / 255.0


class ToVariable:

    def __call__(self, X):
        return Variable(X)


class Normalize:

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, X):
        return (X - self.mean[:, None, None]) / self.std[:, None, None]


class RandomHorizontalFlip:

    def __init__(self, p):
        self.p = p

    def __call__(self, X):

        if np.random.rand() < self.p:
            X = np.flip(X, axis=2).copy()

        return X


class RandomCrop:

    def __init__(self, image_size, padding=2):
        self.image_size = image_size
        self.padding = padding

    def __call__(self, X):
        X = np.pad(
            X,
            ((0, 0), (self.padding, self.padding), (self.padding, self.padding)),
            mode="constant",
        )
        left = np.random.randint(0, 2 * self.padding)
        top = np.random.randint(0, 2 * self.padding)

        return X[:, top : top + self.image_size, left : left + self.image_size].copy()


class ComposeTransforms:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, X):

        for t in self.transforms:
            X = t(X)

        return X
