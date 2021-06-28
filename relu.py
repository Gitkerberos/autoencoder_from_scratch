from module import Module
import numpy as np

class Relu(Module):

    def __init__(self):
        Module.__init__(self)
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        pass

    def forward(self, X):
        # print("----------------------- forward relu -----------------------")
        return np.where(X<0, 0, X)

    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        # print("----------------------- backward delta relu -----------------------")
        return np.where(input >= 0, delta, 0)


############################### tests #####################################

# passed
def test_relu():
    m = Relu()
    print(m.forward(np.array([ [[1,2,3,-1,2,-3], [5,0,-6,-7,-8,2]] ])))
    print( m.backward_delta( np.array([ [[1,2,3,-1,2,-3], [5,0,-6,-7,-8,2]] ]), np.array([[1,2,2,2,2,3],[1,2,2,2,2,3]]) ) )
    print()

def test_backward_delta():
    m = Relu()
    print(m.backward_delta(np.array([[[1, 2, 3], [-1, 2, -3]], [[5, 0, -6], [-7, -8, 2]]]),
                           np.array([[[1, 2, 2], [2, 2, 3]], [[5, 5, 2], [2, 2, 3]]])))
    print()


##########################################################################
