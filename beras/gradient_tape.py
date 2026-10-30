from collections import defaultdict

from beras.core import Diffable, Tensor
import numpy as np
class GradientTape:

    def __init__(self):
        # Dictionary mapping the object id of an output Tensor to the Diffable layer it was produced from.
        self.previous_layers: defaultdict[int, Diffable | None] = defaultdict(lambda: None)

    def __enter__(self):
        # When tape scope is entered, all Diffables will point to this tape.
        if Diffable.gradient_tape is not None:
            raise RuntimeError("Cannot nest gradient tape scopes.")

        Diffable.gradient_tape = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # When tape scope is exited, all Diffables will no longer point to this tape.
        Diffable.gradient_tape = None

    def gradient(self, target: Tensor, sources: list[Tensor]) -> list[Tensor]:
        """
        Computes the gradient of the target tensor with respect to the sources.

        :param target: the tensor to compute the gradient of, typically loss output
        :param sources: the list of tensors to compute the gradient with respect to
        In order to use tensors as keys to the dictionary, use the python built-in ID function here: https://docs.python.org/3/library/functions.html#id.
        """

        ### TODO: Populate the grads dictionary with {weight_id, weight_gradient} pairs and return the gradients of the sources

        queue = [target]                    ## Live queue; will be used to propagate backwards via breadth-first-search.
        grads = defaultdict(lambda: None)   ## Grads to be recorded. Initialize to None. Note: stores {id: list[gradients]}
        # Use id(tensor) to get the object id of a tensor object.
        # in the end, your grads dictionary should have the following structure:
        # {id(tensor): [gradient]}

        # What tensor and what gradient is for you to implement!
        # compose_input_gradients and compose_weight_gradients are methods that will be helpful

        #initially None
        grads[id(target)] = [None]

        while queue:
            current_tensor = queue.pop(0)

            #finding layer that produced this current tensor
            layer = self.previous_layers[id(current_tensor)]
            if layer is None:
                continue

            upstream = grads[id(current_tensor)] #get upstream gradient with this tensor

            #gradients from layers parameters
            weight_grads = layer.compose_weight_gradients(upstream)

            for weight, grad in zip(layer.weights, weight_grads):
                if grads[id(weight)] is None:
                    grads[id(weight)] = []

                grads[id(weight)].append(grad)

            #gradients with respect to layers inputs
            input_grads = layer.compose_input_gradients(upstream)

            for layer_input, grad in zip(layer.inputs, input_grads):
                if grads[id(layer_input)] is None:
                    grads[id(layer_input)] = []

                grads[id(layer_input)].append(grad)

                #continue going back if input was from another layer
                if self.previous_layers[id(layer_input)] is not None:
                    queue.append(layer_input)

        #return in same order as sources
        result = []
        for source in sources:
            source_grads = grads[id(source)]
            if source_grads is None:
                result.append(None)
            else:
                result.append(Tensor(np.sum(source_grads, axis=0)))

        return result