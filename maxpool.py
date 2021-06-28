import numpy as np

from module import Module


class MaxPool1D(Module):
    def __init__(self, k_size, stride):
        Module.__init__(self)
        self.k_size = k_size
        self.stride = stride


    def forward(self, X):
        """
        X : batch de x exemples de taille d * c
        """
        # print("     -------- forward maxpool1d ---------")
        res = []
        k = self.k_size
        s = self.stride
        for x in X :
            resv = []
            xsize = x.shape[0]
            for j in range(x.shape[1]):
                resh = []
                for i in range(0,xsize, s):
                    if i+k <= xsize:
                        a = np.max(x[i:i+k,j])
                        resh.append( a )
                resv.append(resh)
            resv = np.transpose(resv)
            res.append(resv)
        return np.array(res)


    def backward_delta(self, input, delta):
        # print("     -------- backward_delta maxpool ---------      ")
        res = np.zeros((input.shape))
        for x in range(input.shape[0]):
            for c in range(input.shape[2]):
                for i in range(delta.shape[1]):
                    idx = (np.where(input[x,i*self.stride:i*self.stride+self.k_size,c]==np.amax(input[x,i*self.stride:i*self.stride+self.k_size,c]))[0][0])
                    res[x,i*self.stride+idx,c]=delta[x,i,c]
        return res


    def zero_grad(self):
        pass

    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_update_gradient(self, input, delta):
        pass



##################################################### test #############################################################

# passed
def test_forward():
    m = MaxPool1D(4,4)
    input = np.array([[[1, 2, 3], [2, 3, 4], [1, 2, 1], [2, 2, 3], [1, 4, 2], [3, 2, 1], [3, 2, 2], [1, 3, 2],
                       [1, 2, 3], [2, 3, 4], [1, 2, 1], [2, 2, 3], [1, 4, 2], [3, 2, 1], [3, 2, 2], [1, 3, 2]],
                      [[4, 4, 1], [5, 2, 3], [1, 2, 4], [0, 2, 5], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4],
                       [1, 2, 3], [2, 3, 4], [1, 2, 1], [2, 2, 3], [1, 4, 2], [3, 2, 1], [3, 2, 2], [1, 3, 2]]])
    m.forward( input )

# passed
def test_backward_delta():
    m = MaxPool1D(4,4)
    delta = np.array([ [[1,2,3], [9,10,11]], [[1,2,3], [9,10,11]] ])
    input = np.array([[[1,2,3],[2,2,3],[3,1,2],[4,2,3],[5,1,2],[6,1,3],[7,1,2],[8,2,4]],[[9,1,8],[10,2,3],[11,2,5],[12,1,2],[13,2,4],[14,1,2],[15,1,2],[16,1,2]]])
    m.backward_delta(input, delta)


########################################################################################################################

