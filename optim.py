from loss import MSELoss
from module import Linear
from sequentiel import Sequentiel
from activation import TanH, Sigmoide
import numpy as np

class Optim:
    def __init__(self, net, loss, eps=1e-3):
        self.net = net
        self.loss = loss
        self.eps = eps
        self.loss_value = 0
        self.output = None

    def step(self, batch_x, batch_y):
        self.net.zeroGrad()
        self.output = self.net.forward(batch_x) # forward du réseau
        self.loss_value = self.loss.forward(batch_y, self.output).mean() # calcule du cout 
        delta_loss = self.loss.backward(batch_y, self.output) # calcule du delta par rapport à la fonction Loss
        self.net.backward(delta_loss) # calcule du gradient du réseau
        self.net.updateParameters(self.eps) # mise à jour des paramètres de chaque couche du réseau

# Bloc de Fonction SGD

def creer_mini_batches(X, y, batch_size, shuffle=True):
    """ Fonction utile pour le SGD (Stochastic Gradient Descent). Cette fonction permet la création de Mini_batch """
    mini_batches = []
    data = np.hstack((X, y))
    if shuffle:
        np.random.shuffle(data)
    n_minibatches = data.shape[0] // batch_size
    i = 0
  
    for i in range(n_minibatches + 1):
        mini_batch = data[i * batch_size:(i + 1)*batch_size, :]
        X_mini = mini_batch[:, :X.shape[1]]
        Y_mini = mini_batch[:, X.shape[1]:].reshape((-1, y.shape[1]))
        mini_batches.append((X_mini, Y_mini))
    if data.shape[0] % batch_size != 0:
        mini_batch = data[i * batch_size:data.shape[0]]
        X_mini = mini_batch[:, :X.shape[1]]
        Y_mini = mini_batch[:, X.shape[1]:].reshape((-1, y.shape[1]))
        mini_batches.append((X_mini, Y_mini))
    return mini_batches

def sgd(net, fLoss, datax, datay, batch_size = 5, max_iter=10):
    """ Fonction implémentant le stochastique gradient descente (SGD)"""
    allbatch = creer_mini_batches(datax, datay, batch_size)
    opt = Optim(net, fLoss)
    for _ in range(max_iter):
        for batch_x,batch_y in allbatch:
            opt.step(batch_x, batch_y)


################################################## tests ###############################################################


# passed
def testOptim():
    """
    tests Optim
    """

    datax = np.random.randn(20, 10)
    datay = np.random.choice([-1, 1], 20, replace=True)
    datay = datay.reshape(-1, 1)
    print(datay.shape)

    net = Sequentiel([Linear(10,3),TanH(),Linear(3,2), Sigmoide(),Linear(2,1), TanH()])
    loss = MSELoss()
    opt = Optim(net, loss)

    opt.step(datax, datay)

    print("new parameters")
    for param in net.parameters():
        print("module ",param)


def testSGD():
    print("SGD test")
    datax = np.random.randn(40, 10)
    datay = np.random.choice([-1, 1], 40, replace=True)
    datay = datay.reshape(-1, 1)
    print(datay.shape)

    net = Sequentiel([Linear(10,3),TanH(),Linear(3,2), Sigmoide(),Linear(2,1), TanH()])
    loss = MSELoss()
    sgd(net, loss, datax, datay, batch_size=10)

    print("new parameters")
    for param in net.parameters():
        print("module ",param)
########################################################################################################################

