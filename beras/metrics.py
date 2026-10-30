import numpy as np

from beras.core import Callable


class CategoricalAccuracy(Callable):
    def forward(self, probs, labels):
        ## TODO: Compute and return the categorical accuracy of your model
        ## given the output probabilities and true labels.
        ## HINT: Argmax + boolean mask via '=='
        predicted = np.argmax(probs, axis=1)
        actual = np.argmax(labels, axis=1)
        correct = predicted == actual
        return np.mean(correct)
