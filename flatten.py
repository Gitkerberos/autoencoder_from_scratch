import numpy as np

from module import Module


class Flatten(Module):
    def __init__(self):
        Module.__init__(self)

    def forward(self,X):
        # print("-------------- forward flatten -----------------")
        return X.reshape( (X.shape[0], -1) )

    def zero_grad(self):
        pass

    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        # print("-------------- backward delta flatten -------------------")
        return delta.reshape(input.shape)



#################################################### tests #############################################################

# passed
def test_flatten():
    m = Flatten()
    print(m.forward(np.array([ [[1,2,3],[4,5,6]], [[7,8,9],[10,11,12]] ])))
    print(m.backward_delta(np.array([ [[1,2,3],[4,5,6]], [[7,8,9],[10,11,12]] ]), np.array([[1,2,3,1,2,3],[1,5,2,1,4,9]])))

########################################################################################################################
