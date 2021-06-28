import numpy as np


class Loss(object):
    def forward(self, y, yhat):
        pass

    def backward(self, y, yhat):
        pass

class BCE(Loss):
    def __init__(self):
        Loss.__init__(self)

    def forward(self, y, yhat):
        assert (y.shape==yhat.shape), self.__class__.__name__+"[Forward]: Les y et yhat n'ont pas les mêmes dimensions."
        return -(
                y*np.maximum(-100, np.log(yhat))+(1-y)*np.maximum(-100, np.log(1-yhat))
            )
        
    def backward(self, y, yhat):
        assert (y.shape==yhat.shape), self.__class__.__name__+"[Backward]: Les y et yhat n'ont pas les mêmes dimensions."
        return -(y/np.maximum(np.exp(-100),yhat)+(-1+y)/np.maximum(np.exp(-100), 1-yhat)) # prise en compte de yhat tend vers 0 et 1-yhat tend vers 0


class MSELoss(Loss):
    def __init__(self):
        Loss.__init__(self)
    
    def forward(self, y, yhat):
        assert (y.shape==yhat.shape), self.__class__.__name__+"[Forward]: Les y et yhat n'ont pas les mêmes dimensions."
        return np.sum( np.power(np.subtract(y, yhat), 2), axis=1 )

    # test ok
    def backward(self, y, yhat):
        result = np.full(y.shape, 0)
        for i in range(y.shape[1]):
            result[:,i] = np.multiply(np.subtract(y[:,i], yhat[:,i]), -2)
        return result


class LogSoftMax(Loss):
    def __init__(self):
        Loss.__init__(self)

    def forward(self, y, yhat):
        # print("-------- forward loss ---------")
        
        C = np.max(yhat, axis=1)
        return - yhat[np.arange(y.size), y] + C + np.log(np.exp(yhat-C.reshape(-1, 1)).sum(axis=1))


    def backward(self, y, yhat):
        # print("-------- backward loss ---------")
        encode_hot = np.zeros((y.size, yhat.shape[1]))
        encode_hot[np.arange(y.size), y] = 1
        
        C = np.max(yhat, axis=1, keepdims=True)
        exps = np.exp(yhat-C)

        return -encode_hot + exps/np.repeat(exps.sum(axis=1, keepdims=True), yhat.shape[1], axis=1)
        

################################################## tests ###############################################################

################################################## MSELoss ###############################################################

# passed
def test_unitaire_forward():
    objet = MSELoss()
    print(objet.forward(np.array([[1,2,3,4,5],[2,2,3,4,1]]), np.array([[2,2,4,2,3], [4,2,1,7,8]])))


# passed
def test_unitaire_backward():
    objet = MSELoss()
    print(objet.backward(np.array([[1,2,3,4,5],[2,2,3,4,1]]), np.array([[2,2,4,2,3], [4,2,1,7,8]])))

################################################## LogSoftMax ###############################################################

# passed
def test_unitaire_backward_logsm():
    objet = LogSoftMax()
    print(objet.backward(np.array([[1,2,3,4,5],[2,2,3,4,1]]), np.array([[2,2,4,2,3], [4,2,1,7,8]])))

# passed
def test_unitaire_forward_logsm():
    objet = LogSoftMax()
    print(objet.forward(np.array([[1,2,3,4,5],[2,2,3,4,1]]), np.array([[2,2,4,2,3], [4,2,1,7,8]])))

################################################## BCE ###############################################################

# y dans ce cas est censée être binaire (soit 0 soit 1)

# passed
def test_unitaire_backward_bce():
    objet = BCE()
    print(objet.backward(np.array([[0,1,1,0,1],[0,1,0,1,1]]), np.array([[0.2,0.5,0.1,0.8,0.2], [0.34,0.2,0.01,0.7,0.8]])))

# passed
def test_unitaire_forward_bce():
    objet = BCE()
    print(objet.forward(np.array([[0,1,1,0,1],[0,1,0,1,1]]), np.array([[0.2,0.5,0.1,0.8,0.2], [0.34,0.2,0.01,0.7,0.8]])))


########################################################################################################################


