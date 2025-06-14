class Node:
    def __init__(self, propagate_grad=False, keep_grad=False):

        self.grad = None
        self.keep_grad = keep_grad
        self.propagate_grad = propagate_grad

    def propagate(self):

        # Obtain a list of input gradients, corresponding to the list of inputs.
        input_grads = self.backward_fn(
            params=self.backward_fn_params, upstream=self.grad
        )

        # Accumulate grads in input_nodes.
        for input, input_grad in zip(self.input_nodes, input_grads):
            if input.grad is None:
                input.grad = input_grad
            else:
                input.grad += input_grad

        # Disconnect the node from the graph.
        if self.propagate_grad:
            del self.input_nodes
            del self.backward_fn_params
            del self.backward_fn
            del self.topo_visited
            self.propagate_grad = False

        if not self.keep_grad:
            self.grad = None

    # Connects self to computation graph. Used by forward method of a function/module.
    def connect(self, input_nodes, backward_fn, backward_fn_params):
        self.propagate_grad = True if len(input_nodes) > 0 else False

        if self.propagate_grad:
            self.input_nodes = input_nodes
            self.backward_fn_params = backward_fn_params
            self.backward_fn = backward_fn
            self.topo_visited = False

    # The calling node must have backward_fn that accepts None as upstream. The Variable that this node represents must be a scalar.
    def backward(self):

        self.ordering = []
        toposort_nodes(self, self.ordering)

        while self.ordering:
            self.ordering.pop().propagate()
        del self.ordering

    def clear_grad(self):
        self.grad = None


def toposort_nodes(node, ordering):
    for input_node in node.input_nodes:
        if not node.topo_visited and input_node.propagate_grad:
            toposort_nodes(input_node, ordering)

    node.topo_visited = True
    ordering.append(node)
