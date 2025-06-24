import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from dl import Module, Variable


class MaxPool(Module):
    def __init__(self, K=2, stride=1):
        super().__init__()

        self.K = K
        self.stride = stride

    def forward(self, X):
        N, C_in, H, W = X.data.shape

        X_windows = sliding_window_view(
            X.data, window_shape=(1, self.K, self.K), axis=(1, 2, 3)
        )
        X_windows = X_windows[:, :, :: self.stride, :: self.stride]
        X_rows = np.reshape(X_windows, shape=(-1, 1 * self.K * self.K))
        patch_flat_maxes = np.max(X_rows, axis=1)

        H_out = (H - self.K) // self.stride + 1
        W_out = (W - self.K) // self.stride + 1
        Y_data = np.reshape(patch_flat_maxes, shape=(N, C_in, H_out, W_out))
        Y = Variable(Y_data)

        input_nodes = []
        backward_fn_params = {}

        if X.node.keep_grad or X.node.propagate_grad:
            input_nodes.append(X.node)

            backward_fn_params["X"] = X.data
            backward_fn_params["K"] = self.K
            backward_fn_params["stride"] = self.stride

        Y.node.connect(input_nodes, max_pool_backward, backward_fn_params)

        return Y


def max_pool_backward(params, dY=None):
    X = params["X"]
    K = params["K"]
    stride = params["stride"]

    N, C_in, H, W = X.shape
    X_windows = sliding_window_view(X, window_shape=(1, K, K), axis=(1, 2, 3))
    X_windows = X_windows[:, :, ::stride, ::stride]

    H_out = (H - K) // stride + 1
    W_out = (W - K) // stride + 1

    X_rows = np.reshape(X_windows, shape=(-1, 1 * K * K))

    # For each row, recover the input index corresponding to the argmax of that row.
    patch_flat_indices = np.argmax(X_rows, axis=1)
    patch_indices = np.unravel_index(
        patch_flat_indices, shape=(K, K)
    )  # 2d index relative to the window.

    offset_W = np.tile(np.arange(W_out) * stride, H_out)
    offset_W = np.tile(offset_W, C_in)
    offset_W = np.tile(offset_W, N)

    offset_H = np.repeat(np.arange(H_out) * stride, W_out)
    offset_H = np.tile(offset_H, C_in)
    offset_H = np.tile(offset_H, N)

    offset_C_in = np.repeat(np.arange(C_in), H_out * W_out)
    offset_C_in = np.tile(offset_C_in, N)

    offset_N = np.repeat(np.arange(N), C_in * H_out * W_out)

    indices_W = patch_indices[1] + offset_W
    indices_H = patch_indices[0] + offset_H
    indices_C_in = offset_C_in
    indices_N = offset_N

    # Indices is now a tuple of 4 arrays, each of length N * C_in * H_out * W_out, which
    # is the number of rows in X_rows. For each row, its argmax has a corresponding
    # 4d index in the input shape, which is exactly what indices contains.
    indices = (indices_N, indices_C_in, indices_H, indices_W)

    dY = np.reshape(dY, shape=(-1))

    G_XY = np.zeros(X.shape)
    np.add.at(G_XY, indices, dY)

    return [G_XY]
