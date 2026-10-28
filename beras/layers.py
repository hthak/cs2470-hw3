import numpy as np

from typing import Literal
from beras.core import Diffable, Variable, Tensor

DENSE_INITIALIZERS = Literal["zero", "normal", "xavier", "kaiming", "xavier uniform", "kaiming uniform"]

class Dense(Diffable):

    def __init__(self, input_size, output_size, initializer: DENSE_INITIALIZERS = "normal"):
        self.w, self.b = self._initialize_weight(initializer, input_size, output_size)

    @property
    def weights(self) -> list[Tensor]:
        return self.w, self.b

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for a dense layer! Refer to lecture slides for how this is computed.
        """
        return Tensor(x @ self.w + self.b)

    def get_input_gradients(self) -> list[Tensor]:
        return [Tensor(self.w.T)]

    def get_weight_gradients(self) -> list[Tensor]:
        x = self.inputs[0]
        weight_gradients = np.expand_dims(x, axis=-1)
        bias_gradients = np.ones_like(self.b) #this will be all constants

        return [Tensor(weight_gradients), Tensor(bias_gradients)]

    @staticmethod
    def _initialize_weight(initializer, input_size, output_size) -> tuple[Variable, Variable]:
        """
        Initializes the values of the weights and biases. The bias weights should always start at zero.
        However, the weights should follow the given distribution defined by the initializer parameter
        (zero, normal, xavier, or kaiming). You can do this with an if statement
        cycling through each option!

        Details on each weight initialization option:
            - Zero: Weights and biases contain only 0's. Generally a bad idea since the gradient update
            will be the same for each weight so all weights will have the same values.
            - Normal: Weights are initialized according to a normal distribution.
            - Xavier: Goal is to initialize the weights so that the variance of the activations are the
            same across every layer. This helps to prevent exploding or vanishing gradients. Typically
            works better for layers with tanh or sigmoid activation.
            - Kaiming: Similar purpose as Xavier initialization. Typically works better for layers
            with ReLU activation.
        """

        initializer = initializer.lower()
        assert initializer in (
            "zero",
            "normal",
            "xavier",
            "kaiming",
        ), f"Unknown dense weight initialization strategy '{initializer}' requested"

        if initializer == "zero":
            w = np.zeros((input_size, output_size))
        elif initializer == "normal":
            w = np.random.normal(loc=0, scale=1,size=(input_size, output_size))
        elif initializer == "xavier":
            std = np.sqrt(2 / (input_size+output_size))
            w = np.random.normal(loc=0, scale=1,size=(input_size, output_size))
        elif initializer == "kaiming":
            std = np.sqrt(2 / input_size)

            w = np.random.normal(
                loc=0,
                scale=std,
                size=(input_size, output_size)
            )
        b = np.zeros(output_size) #since biases always start at 0

        return Variable(w), Variable(b) #not using tensor
