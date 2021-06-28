from loss import Loss
from module import Module
import numpy as np



class SoftMax(Module):
    def __init__(self):
        Module.__init__(self)
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        pass

    def forward(self, X):
        #print("-------- forward loss ---------")
        return np.divide(
            np.exp(X), np.transpose( np.tile(
                np.sum( np.exp(X), axis=1 ), (X.shape[1], 1)
            ) )
        )

    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_update_gradient(self, input, delta):
        pass


    def backward_delta(self, input, delta):
        a = np.divide(
            np.exp(input),
            np.transpose( np.tile(
                np.sum( np.exp(input), axis=1),
                (input.shape[1], 1)
            ) )
        )

        res = []
        for x in a:
            b = np.tile(x, (input.shape[1], 1)).reshape((-1,))
            c = np.add(1, np.multiply(-1, np.transpose( np.tile(x,input.shape[1]) ).reshape((-1,))))
            d = np.multiply( b,c )
            res.append( d.reshape((input.shape[1], delta.shape[1])) )
        return np.array(res)



################################################## tests ###############################################################

# passed
def test_unitaire_forward_sm():
    objet = SoftMax()
    print(objet.forward(np.array([[2,2,4,2,3], [4,2,1,7,8]])))

# passed
def test_backward_delta_sm():
    objet = SoftMax()
    print(objet.backward_delta(np.array([[2,2,4,2,3], [4,2,1,7,8]]), np.array([ [2,2,1,2,3], [1,2,2,1,2] ])))

########################################################################################################################

