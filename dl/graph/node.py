# Notes
# Variables/Parameters have references to Nodes, but not the other way around. They can be garbage
# collected while the node continues to exist (and hold the gradient) in the computation graph.
# This design allows users to choose which variables should be saved. I may implement a save output option
# For each module.


class Node:
    def __init__(self, propagate_grad=False, keep_grad=False):
        self.grad = None

        # The difference between keep_grad and propagate_grad is important:
        # 1. propagate_grad is True iff there is an input node for which (keep_grad or propagate_grad) evaluates to True.
        #    This is handled automatically by the forward pass of the module that created the node. Only modify propagate_grad yourself if you have a specific reason.
        # 2. keep_grad is specified by the user and is True iff the user wants the gradient of the node to remain stored in .grad after backprop.
        #    Otherwise, .grad is deleted to free memory.

        self.keep_grad = keep_grad
        self.propagate_grad = propagate_grad

    # Calling this method will propagate self.grad and accumulate into .grad for input nodes.
    # This method should only be called if propagate_grad is True
    # This method should only be called when all parents of self (that can be reached from the root node) have already called .backward.
    # This method is called automatically. Users should only call it if they have a specific reason to.
    def propagate(self):
        # obtain a list of input gradients, corresponding to the list of inputs
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

    # Connects self to computation graph. Used by forward method of a module.
    def connect(self, input_nodes, backward_fn, backward_fn_params):
        self.propagate_grad = True if len(input_nodes) > 0 else False

        if self.propagate_grad:
            self.input_nodes = input_nodes
            self.backward_fn_params = backward_fn_params
            self.backward_fn = backward_fn
            self.topo_visited = False

    # Self must have backward_fn that accepts None as upstream (I.e., the variable represented by self must be a scalar).
    def backward(self):
        # 1. Use topological sort to create an order of nodes.
        # The order must satisfy:
        #   1. Only descendants of self.node with propagate == True are in the order
        #   2. All parent nodes of a child appear after the child in the order

        self.ordering = []
        toposort_nodes(self, self.ordering)

        # 2. Backwards in reverse of order
        while self.ordering:
            self.ordering.pop().propagate()
        del self.ordering

    def clear_grad(self):
        self.grad = None


# Helper method for backward: creates a valid order for nodes to propagate.
def toposort_nodes(node, ordering):
    for input_node in node.input_nodes:
        if not node.topo_visited and input_node.propagate_grad:
            toposort_nodes(input_node, ordering)

    node.topo_visited = True
    ordering.append(node)
