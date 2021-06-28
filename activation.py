import numpy as np
from module import Module


class TanH(Module):
    """ Le module d'activation n'a pas de paramètre """

    def __init__(self):
        Module.__init__(self)

    def zero_grad(self):
        self._gradient = 0

    def forward(self, X):
        return np.tanh(X)

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        return (1 - np.tanh(input) ** 2) * delta

    def update_parameters(self, gradient_step=1e-3):
        # il n'existe pas de paramètre pour le module d'activation
        pass


class Sigmoide(Module):
    def __init__(self):
        Module.__init__(self)

    def zero_grad(self):
        self._gradient = 0

    def forward(self, X):
        return (1 / (1 + np.exp(-X)))

    def backward_update_gradient(self, input, delta):
        # là on n'a pas de paramètres avec les modules d'activation du coup le gradient de Z^h=sig(Z^(h-1)) par rapport à W^h est de 0
        pass

    def backward_delta(self, input, delta):
        sig = (1 / (1 + np.exp(-input)))
        return sig * (1 - sig) * delta

    def update_parameters(self, gradient_step=1e-3):
        # il n'existe pas de paramètre pour le module d'activation
        pass
