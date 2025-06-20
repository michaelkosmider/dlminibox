import numpy as np
from dl import Parameter, Variable, Module
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
        # Reshape X and W to compute convolution as a matrix multiplication. Also add padding beforehand.
        X_padded = pad_and_dilate(X.data, self.padding)
        X_rows = im2row(X_padded, self.C_in, self.K, self.stride)

        W_columns = np.reshape(
            self.W.data, shape=(self.C_out, self.C_in * self.K * self.K)
        ).T

        # Perform the matrix multiplication and reshape.
        Y_mat = np.matmul(X_rows, W_columns)

        N_batch, _, H_in, W_in = X.data.shape
        H_out = (H_in + 2 * self.padding - self.K) // self.stride + 1
        W_out = (W_in + 2 * self.padding - self.K) // self.stride + 1
        Y_data = np.reshape(Y_mat, shape=(N_batch, H_out, W_out, self.C_out))
        Y_data = np.transpose(Y_data, axes=(0, 3, 1, 2))

        Y = Variable(Y_data)

        # Connect node in computation graph.
        input_nodes = []
        backward_fn_params = {}

        if self.W.node.propagate_grad or self.W.node.keep_grad:
            input_nodes.append(self.W.node)

            backward_fn_params["X"] = X.data
            backward_fn_params["C_in"] = self.C_in
            backward_fn_params["K"] = self.K
            backward_fn_params["stride"] = self.stride
            backward_fn_params["padding"] = self.padding

        if X.node.propagate_grad or X.node.keep_grad:
            input_nodes.append(X.node)

            backward_fn_params["W"] = self.W.data
            backward_fn_params["stride"] = self.stride
            backward_fn_params["padding"] = self.padding
            backward_fn_params["X_shape"] = X.data.shape

        Y.node.connect(input_nodes, convolution_backward, backward_fn_params)

        return Y


def convolution_backward(params, dY=None):
    grads = []

    if "X" in params:
        X = params["X"]
        C_in = params["C_in"]
        K = params["K"]
        stride = params["stride"]
        padding = params["padding"]

        # We store X and recompute X_rows because it's too expensive to keep in memory.
        X_padded = pad_and_dilate(X, padding=padding)
        X_rows = im2row(X_padded, C_in, K, stride)

        # Reshape upstream to match shape of matrix multiplication output (not module output).
        C_out = dY.shape[1]
        dY_columns = np.transpose(dY, axes=(0, 2, 3, 1))
        dY_columns = np.reshape(dY_columns, shape=(-1, C_out))

        # Compute G_W
        G_W_rows = np.matmul(X_rows.T, dY_columns).T
        G_W = np.reshape(G_W_rows, shape=(C_out, C_in, K, K))

        grads.append(G_W)

    if "W" in params:
        W = params["W"]
        stride = params["stride"]
        padding = params["padding"]
        X_shape = params["X_shape"]

        # Initialize G_XY to the correct shape of zeros.
        G_XY = np.zeros(shape=X_shape, dtype=dY.dtype)

        # Flip and flatten the kernel.
        C_out, C_in, K, _ = W.shape
        W_flipped = np.transpose(np.flip(W, axis=(-2, -1)), axes=(1, 0, 2, 3))
        W_flipped_columns = np.reshape(W_flipped, shape=(C_in, C_out * K * K)).T

        # Dialate, pad, and reshape upstream gradient dY.
        dY_dil = pad_and_dilate(dY, padding=K - 1, dilation=stride - 1)

        N_batch, _, H_in, W_in = X_shape

        # Perform convolution with flipped kernel and modified dY.
        span_H = dY_dil.shape[2] - K + 1
        span_W = dY_dil.shape[3] - K + 1

        dY_dil_rows = im2row(dY_dil, C_out, K, stride=1)

        G_XY_mat = np.matmul(dY_dil_rows, W_flipped_columns)

        G_XY_span = np.reshape(G_XY_mat, shape=(N_batch, span_H, span_W, C_in))
        G_XY_span = np.transpose(G_XY_span, axes=(0, 3, 1, 2))
        G_XY_span = G_XY_span[:, :, padding : padding + H_in, padding : padding + W_in]
        _, _, H_out, W_out = G_XY_span.shape

        G_XY[:, :, :H_out, :W_out] = G_XY_span

        grads.append(G_XY)

    return grads


def im2row(X, C_in, K, stride):
    # Convert each image patch to a row.
    X_windows = sliding_window_view(
        X, window_shape=(C_in, K, K), axis=(1, 2, 3)
    )  # Shape: (N_batch, 1, H_out, W_out, C_in, K, K)
    X_windows = np.squeeze(X_windows, 1)
    X_windows = X_windows[:, ::stride, ::stride]

    X_rows = np.reshape(
        X_windows, shape=(-1, C_in * K * K)
    )  # Memory intensive step: every window in the image is now explicitly copied into its own row.

    return X_rows


def pad_and_dilate(X, padding=0, dilation=0):
    N, C_in, H, W = X.shape

    H_new = (H - 1) * (dilation + 1) + 1 + 2 * padding
    W_new = (W - 1) * (dilation + 1) + 1 + 2 * padding

    X_new = np.zeros(shape=(N, C_in, H_new, W_new), dtype=X.dtype)
    X_new[
        :,
        :,
        padding : padding + (H - 1) * (dilation + 1) + 1 : (dilation + 1),
        padding : padding + (W - 1) * (dilation + 1) + 1 : (dilation + 1),
    ] = X

    return X_new
