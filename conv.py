from module import Module
import numpy as np



class Conv1D(Module):
    def __init__(self, k_size, chan_in, chan_out, stride):
        Module.__init__(self)
        self.k_size = k_size
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.stride = stride
        self._parameters = 0.02 * (np.random.rand(chan_out, k_size, chan_in) - 0.5)  # initialise parameters w
        #self._parameters = 0.05 * np.ones((chan_out, k_size, chan_in))
        if chan_in == 1 :
            self._parameters = self._parameters.reshape((chan_out, k_size))
        self._gradient = []  # initialise parameters' w gradient



    def zero_grad(self):
        self._gradient = np.zeros(self._parameters.shape)



    def forward(self, X):
        # print("----------------------- forward conv -----------------------")
        res = []
        for x in X:
            res_x = []
            for filter in self._parameters:
                for i in range(0, x.shape[0], self.stride):
                    x_in = x[i:i+self.stride]
                    res_x.append(sum(x_in*filter))
            res_x = np.array(res_x)
            res_x = res_x.reshape((self._parameters.shape[0], -1)) # nbr filters
            res_x = np.transpose(res_x)
            res.append(res_x)
        return np.array(res)



    def update_parameters(self, gradient_step=1e-3):
        self._parameters -= gradient_step * self._gradient * self._parameters



    def backward_update_gradient(self, input, delta):
        self.zero_grad()
        for x in range(input.shape[0]):
            for f in range(self._parameters.shape[0]):
                aux = np.transpose(input[x].reshape((input.shape[1]//self.stride, -1)))
                #print("aux : ", aux)
                self._gradient[f] += aux@delta[x,:,f]
                #print("gradient : ", aux@delta[x,:,f])



    def backward_delta(self, input, delta):
        # print("----------------------- backward delta conv -----------------------")
        res = []
        for x in range(input.shape[0]):
            res_x = []
            for f in range(self._parameters.shape[0]): # for each filter
                w = self._parameters[f] # the filter
                d = delta[x,:,f] # first column of x's delta corresponding to filter f
                for d_ in d :
                    res_x.extend(np.multiply(w, d_))
            res_x = np.array(res_x).reshape((self.chan_out, -1))
            res_x = np.transpose(np.array(res_x))
            res.append(res_x)
        res = np.array(res)
        return res


###################################################### test ############################################################

# passed
def test_forward():
    m = Conv1D(4,1,3,4) # k_size, chan_in, chan_out, stride
    print("initial parameters : ", m._parameters)
    #input = np.array([[[1,2,3], [2,3,4],[1,2,1], [2,2,3],[1,4,2], [3,2,1],[3,2,2], [1,3,2], [1,2,3], [2,3,4],[1,2,1], [2,2,3],[1,4,2], [3,2,1],[3,2,2], [1,3,2]],
        #              [[4,4,1], [5,2,3],[1,2,4], [0,2,5],[1,1,1], [2,2,2],[3,3,3], [4,4,4], [1,2,3], [2,3,4],[1,2,1], [2,2,3],[1,4,2], [3,2,1],[3,2,2], [1,3,2]]])
    input = np.array([[1,2,3,4,5,6,1,2,5,1,3,4,2,1,2,1],[2,1,2,2,4,3,2,5,1,6,2,11,2,3,4,5]])
    #print("input : ", input)
    print("forward : ",m.forward(input))

# passed
def test_backward_delta():
    m = Conv1D(4,1,3,4) # k_size, chan_in, chan_out, stride
    #print("initial parameters : ", m._parameters)
    #input = np.array([[[1, 2, 3], [2, 3, 4], [1, 2, 1], [2, 2, 3], [1, 4, 2], [3, 2, 1], [3, 2, 2], [1, 3, 2],
    #                   [1, 2, 3], [2, 3, 4], [1, 2, 1], [2, 2, 3], [1, 4, 2], [3, 2, 1], [3, 2, 2], [1, 3, 2]],
    #                  [[4, 4, 1], [5, 2, 3], [1, 2, 4], [0, 2, 5], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4],
    #                   [1, 2, 3], [2, 3, 4], [1, 2, 1], [2, 2, 3], [1, 4, 2], [3, 2, 1], [3, 2, 2], [1, 3, 2]]])
    input = np.array([[1,2,3,4,5,6,1,2],[2,1,2,2,4,3,2,5]])
    #print("input : ", input)
    print("backward delta : ", m.backward_delta(input, np.array([[[0.5,0.25,0.125],[0.5,0.25,0.125]], [[0.5,0.25,0.125],[0.5,0.25,0.125]]]) ) )

# passed
def test_backward_gr():
    m = Conv1D(4,1,3,4) # k_size, chan_in, chan_out, stride
    #print("initial parameters : ", m._parameters)
    #input = np.array([[[1, 2, 3], [2, 3, 4], [1, 2, 1], [2, 2, 3], [1, 4, 2], [3, 2, 1], [3, 2, 2], [1, 3, 2],
                 #      [1, 2, 3], [2, 3, 4], [1, 2, 1], [2, 2, 3], [1, 4, 2], [3, 2, 1], [3, 2, 2], [1, 3, 2]],
                #      [[4, 4, 1], [5, 2, 3], [1, 2, 4], [0, 2, 5], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4],
                #       [1, 2, 3], [2, 3, 4], [1, 2, 1], [2, 2, 3], [1, 4, 2], [3, 2, 1], [3, 2, 2], [1, 3, 2]]])
    input = np.array([[1,2,3,4,5,6,1,2],[2,1,2,2,4,3,2,5]])
    #print("input : ", input)
    print("backward gradient : ")
    m.backward_update_gradient( input, np.array([[[0.5,0.25,0.125],[0.5,0.25,0.125]], [[0.5,0.25,0.125],[0.5,0.25,0.125]]]) )
    print(m._gradient)

# passed
def test_updatep():
    m = Conv1D(4, 1, 3, 4)  # k_size, chan_in, chan_out, stride
    #input = np.array([[[1, 2, 3], [2, 3, 4], [1, 2, 1], [2, 2, 3], [1, 4, 2], [3, 2, 1], [3, 2, 2], [1, 3, 2],
              #         [1, 2, 3], [2, 3, 4], [1, 2, 1], [2, 2, 3], [1, 4, 2], [3, 2, 1], [3, 2, 2], [1, 3, 2]],
                #      [[4, 4, 1], [5, 2, 3], [1, 2, 4], [0, 2, 5], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4],
                 #      [1, 2, 3], [2, 3, 4], [1, 2, 1], [2, 2, 3], [1, 4, 2], [3, 2, 1], [3, 2, 2], [1, 3, 2]]])
    input = np.array([[1,2,3,4,5,6,1,2],[2,1,2,2,4,3,2,5]])
    m.backward_update_gradient( input, np.array([[[0.5,0.25,0.125],[0.5,0.25,0.125]], [[0.5,0.25,0.125],[0.5,0.25,0.125]]])  )
    m.update_parameters(0.001)
    print(m._parameters)


########################################################################################################################
