import numpy as np
from loss import MSELoss

class Module(object):
    def __init__(self):
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        ## Annule gradient
        pass

    def forward(self, X):
        ## Calcule la passe forward
        pass

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self._parameters -= gradient_step*self._gradient

    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        pass



class Linear(Module):
    def __init__(self, input, output):
        Module.__init__(self)
        #np.random.seed(1)
        self._parameters = 0.05 * (np.random.rand(input, output) - 0.5)  #2 *(np.random.rand(input,output)-0.5) # initialise parameters w
        self._gradient = np.zeros((input,output)) # initialise parameters' w gradient
        self._bias = np.zeros((1, output))
        self._gradient_bias = np.zeros((1, output))

    def zero_grad(self):
        self._gradient[:] = 0
        self._gradient_bias[:] = 0

    def forward(self, X):
        X = X.reshape(-1, self._gradient.shape[0]) 
        # prise en compte d'un biais
        return X@self._parameters + self._bias


    def backward_update_gradient(self, input, delta):
        """
        Le backward_update_gradient consiste à calculer le gradient du coût par rapport aux paramètres et        l’additionner à la variable _gradient en fonction de l’entrée (input) et des delta de la couche suivante

        Etant linéaire le module, le gradient de Z^h par rapport aux paramètres W^h du module est Z^h-1 étant donné que Z^h = Z^(h-1).W^h + B. delta étant  de taille batch x dim_output et input(Z^(h-1)) est de dimension batch x dim_input. Selon la formule (1) du sujet du projet on a le gradient du coût par rapport aux paramètres qui est: 
        input.T@delta 
        qui va donner un gradient de dimension dim_input x dim_output
        """
        input = input.reshape(-1, self._gradient.shape[0])
        delta = delta.reshape(-1, self._gradient.shape[1])
        assert (input.shape[0]==delta.shape[0]), self.__class__.__name__+"[backward_update_gradient] la taille des batchs ne sont pas les mêmes."
        assert (delta.shape[1] == self._parameters.shape[1]), self.__class__.__name__+"[backward_update_gradient] delta doit avoir la même dimension que la sortie du module. delta: batch x dim_output"

        # reshape input to dim_input x batch
        self._gradient += input.T@delta
        self._gradient_bias += delta.sum(axis=0, keepdims=True)


    def backward_delta(self, input, delta):
        """ Module linéaire donc  Z^h = Z^(h-1).W^h + B et par conséquent le gradient de Z^h par rapport à Z^(h-1) est W^h de dimension dim_input x dim_output et delta est du module suivant est de taille batch x dim_output: le delta du module d'avant serait donc:
        delta@W^h.T avec "T" la transposée de la matrice W^h
        On aura un résulat de taille batch x dim_input. Et ce dim_input serait le dim_output du module d'avant
        """
        input = input.reshape(-1, self._gradient.shape[0])
        delta = delta.reshape(-1, self._gradient.shape[1])
        assert (input.shape[0]==delta.shape[0]), self.__class__.__name__+"[backward] la taille des batchs ne sont pas les mêmes."

        return delta@self._parameters.T
    
    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self._parameters -= gradient_step*self._gradient
        self._bias -= gradient_step*self._gradient_bias


################################################## tests ###############################################################


datax = np.random.randn(20,10)
datay = np.random.choice([-1,1],20,replace=True)
dataymulti = np.random.choice(range(10),20,replace=True)
linear = Linear(10,1)
#sigmoide = Sigmoide()
#softmax = Softmax()
#tanh = Tanh()
#relu = ReLU()
#conv1D = Conv1D(k_size=3,chan_in=1,chan_out=5,stride=2)
#maxpool1D = MaxPool1D(k_size=2,stride=2)
#flatten = Flatten()

mse = MSELoss()
#bce = BCE()
#crossentr = CrossEntropy() #cross entropy avec log softmax


# passed
def test1():
    """
    simple test pour regarder si les reshape marchent correctement. Reseau à 2 couches
    """
    l1 = Linear(5,3)
    l2 = Linear(3,2)
    loss = MSELoss()

    initial_input = np.array([ [1,3,4,1,2], [2,4,5,2,1], [5,1,2,3,4], [4,1,2,1,5], [6,3,1,7,2] ])
    print("\ninitial input : \n", initial_input)

    print("-------------------- forward ---------------------")

    result_l1 = l1.forward( initial_input )
    print( "\n     result_l1\n", result_l1 )
    result_l2 = l2.forward( result_l1 )
    print( "\n     result_l2\n", result_l2 )
    loss_ = loss.forward( result_l2, np.array([ [0,1],[1,0],[1,0], [1,0], [0,1] ]) )
    print( "\n     loss_\n", loss_ )
    grad_loss = loss.backward( np.array([ [0,1],[1,0],[1,0], [1,0], [0,1] ]), result_l2 )
    print( "\n     grad_loss\n", grad_loss )

    print("------------------- backward ---------------------")

    d2 = l2.backward_delta( result_l1, grad_loss )
    print( "\n     d2\n", d2 )
    print("\n     old w for l2 : \n", l2._parameters)
    l2.backward_update_gradient( result_l1,grad_loss )
    print("\n     new w for l2 : \n", l2._parameters)
    print("\n     old w for l1 : \n", l1._parameters)
    l1.backward_update_gradient( initial_input, d2 )
    print("\n     new w for l1 : \n", l1._parameters)

# passed
def test2():
    """
    script de vérification
    """
    linear.zero_grad()
    res_lin = linear.forward(datax)
    res_mse = mse.forward(datay.reshape(-1, 1), res_lin)
    delta_mse = mse.backward(datay.reshape(-1, 1), res_lin)
    linear.backward_update_gradient(datax, delta_mse)
    grad_lin = linear._gradient
    print(grad_lin)
    delta_lin = linear.backward_delta(datax, delta_mse)
    print(delta_lin)


# passed
def test_unitaire_forward():
    l = Linear(3,2)
    print(l.forward(np.array([[2,1,3]])))
    l.backward_update_gradient(np.array([1,2,3]),np.array([1,2]))
    print( "\n", l.backward_delta( np.array([ [1,2,1], [3,4,2], [1,5,6] ]), np.array([1,2]) ) )

# passed
def test_unitaire_backward():
    l2 = Linear(3,2)
    l2.backward_delta( np.array([ [1,2,3], [2,3,4] ]), np.array([ [1,2], [2,3] ]) )



########################################################################################################################

