from loss import MSELoss, Loss
from module import Linear, Module
import numpy as np


class Sequentiel(Module):

    def __init__(self, modules=None):
        """
        :param modules: tableau de modules dans l'ordre de traitement
        """
        Module.__init__(self)
        if modules == None:
            self._modules = []
        else:
            self._modules = modules
        self._inputs = []


    def ajout_modules(self, modules):
        """
        ajout des modules
        :param modules: list des modules à ajouter (si un seul module, il faut l'inclure dans une liste)
        """
        self._modules.extend(modules)


    def forward(self, input):
        self._inputs = [] # Il faut une remise à zéro des input car les paramètres(voir les données) ont changé. Utile pour la fonction sgd 
        self._inputs.extend([input])
        for module in self._modules:
            input = module.forward(input)
            self._inputs.extend([input])
        return np.array(self._inputs[-1]) # le dernier output du réseau qui serait le input du loss


    def backward(self, delta_loss):
        """
        :param delta_loss: delta calculé avec le loss par rapports au labels. 
        """
        list_inputs = self._inputs[:-1]
        inverse_inputs = list_inputs[::-1]
        inverse_modules = self._modules[::-1]
        # self._loss = self._loss_module.forward(labels, inverse_inputs[0])

        ### find delta ###
        # delta = []
        # #d = self._loss_module.backward(labels, inverse_inputs[0])
        # delta.append(delta_loss)
        # for i in range(len(inverse_inputs)):
        #     d = inverse_modules[i].backward_delta(inverse_inputs[i], delta[-1])
        #     delta.append(d)

        # ### update modules' parameters ###
        # #delta = delta[::-1]
        # for i in range(len(inverse_modules)):
        #     inverse_modules[i].backward_update_gradient(inverse_inputs[i], delta[i])
        delta = delta_loss
        for i in range(len(inverse_modules)):
            inverse_modules[i].backward_update_gradient(inverse_inputs[i], delta)
            delta = inverse_modules[i].backward_delta(inverse_inputs[i], delta)

    def updateParameters(self, eps=1e-3):
        for module in self._modules:
            module.update_parameters(gradient_step=eps)
    
    def zeroGrad(self):
        for module in self._modules:
            module.zero_grad()

    def parameters(self):
        parameters = []
        for module in self._modules:
            parameter = module._parameters
            if parameter is not None:
                parameters.append(parameter)
        return parameters




################################################## tests ###############################################################


# passed
def test3():
    """
    tests Sequentiel module
    """
    #initial_input = np.array([[1, 3, 4, 1, 2], [2, 4, 5, 2, 1], [5, 1, 2, 3, 4], [4, 1, 2, 1, 5], [6, 3, 1, 7, 2]])
    #print("\ninitial input : \n", initial_input)

    datax = np.random.randn(20, 10)
    datay = np.random.choice([-1, 1], 20, replace=True)
    datay = datay.reshape(datay.shape[0], 1)
    datay = np.hstack((datay, datay))
    print(datay)

    floss = MSELoss()
    m1 = Linear(10,3)
    m2 = Linear(3,2)
    sequence = Sequentiel([m1,m2])

    print("-------------------- forward ---------------------")
    sequence.zeroGrad()
    output = sequence.forward(datax)

    print("------------------- backward ---------------------")
    #labels =  np.array([ [0,1],[1,0],[1,0], [1,0], [0,1] ])
    floss.forward(datay, output)
    delta_loss = floss.backward(datay, output)
    sequence.backward(delta_loss)
    #sequence.updateParameters()

    print("new parameters : \n", m1._parameters, "\n\n", m2._parameters)


########################################################################################################################

