# coming soon!
import numpy as np
from dl.graph import Node
from dl import Parameter
from dl import Module
from numpy.lib.stride_tricks import sliding_window_view


class Convolution(Module):
    def __init__(self, C_out, C_in, K, stride, padding, param_init="He"):
        super().__init__()

        # Store hyper_parameters for print_module() and forward.
        self.C_in = C_in
        self.C_out = C_out
        self.K = K
        self.stride = stride
        self.padding = padding
        self.param_init = param_init

        # Weights are initialized using He (more types of initialization to be added later).
        if param_init == "He":
            fan_in = C_in * K * K
            std = np.sqrt(2 / fan_in)
            self.W = Parameter(
                np.random.normal(0, std, size=(C_out, C_in, K, K)), keep_grad=True
            )

    def forward(self, X):
        N_batch = X.shape[0]

        # Standard formulas for output size. We need these later.
        H_in = X.shape[2]
        W_in = X.shape[3]
        H_out = (H_in + 2 * self.padding - self.K) // self.stride + 1
        W_out = (W_in + 2 * self.padding - self.K) // self.stride + 1

        # Reshape X and W to compute convolution as a matrix multiplication.
        X_padded = np.pad(
            X,
            (
                (0, 0),
                (0, 0),
                (self.padding, self.padding),
                (self.padding, self.padding),
            ),
        )
        X_rows = im2row(X_padded, self.C_in, self.K, self.stride)

        W_columns = np.reshape(
            self.W, shape=(self.C_out, self.C_in * self.K * self.K)
        ).T

        # Perform the convolution
        Y = np.matmul(X_rows, W_columns)  # Where all the dot products happen.

        Y = np.reshape(Y, shape=(N_batch, H_out, W_out, self.C_out))
        Y = np.transpose(Y, axes=(0, 3, 1, 2))

        return Y


def im2row(X, C_in, K, stride):
    # Convert each image patch to a row.
    X_windows = sliding_window_view(
        X, window_shape=(C_in, K, K), axis=(1, 2, 3)
    )  # Shape: (N_batch, 1, H_out, W_out, C_in, K, K)
    X_windows = np.squeeze(X_windows, 1)
    X_windows = X_windows[:, ::stride, ::stride]

    X_rows = np.reshape(
        X_windows, shape=(-1, C_in * K * K)
    )  # Memory intensive step: every image window is explicitly copied into its own row.

    return X_rows
