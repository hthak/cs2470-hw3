import numpy as np

from beras.core import Diffable, Tensor


class Loss(Diffable):
    @property
    def weights(self) -> list[Tensor]:
        return []

    def get_weight_gradients(self) -> list[Tensor]:
        return []


class MeanSquaredError(Loss):
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return Tensor(np.mean((y_pred-y_true)**2))

    def get_input_gradients(self) -> list[Tensor]:
        y_pred, y_true = self.inputs
        #num_outputs = y_pred.shape[-1]
        pred_grad = 2 * (y_pred - y_true)/y_pred.size
        true_grad = np.zeros_like(y_true) #constant vector
        return [Tensor(pred_grad), Tensor(true_grad)]

class CategoricalCrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        """Categorical cross entropy forward pass!"""
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-(1e-7)) #clamp btwn epsilon and 1-epsilon
        loss = -np.sum(y_true*np.log(y_pred_clipped), axis=1)
        return Tensor(np.mean(loss))

    def get_input_gradients(self):
        """Categorical cross entropy input gradient method!"""
        y_pred, y_true = self.inputs

        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        pred_grad = -y_true / y_pred_clipped
        true_grad = np.zeros_like(y_true) #constant vector

        return [Tensor(pred_grad), Tensor(true_grad)]

