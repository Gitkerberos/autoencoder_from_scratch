from optim import Optim, creer_mini_batches
import numpy as np


class DefaultAutoencoder:
    def __init__(self, net, fLoss, learning_rate=1e-3):
        self.optim = Optim(net, fLoss, learning_rate)

    def train(self, features, max_iter=50, size_per_batch=30):
        """ Apprentissage de notre autoencodeur """
        pass

    def reconstruct(self, data):
        """ Reconstruiction des données en utilisant le réseau qu'on a appris """
        pass

    def getAutoencoder(self):
        return self.optim.net

    def getLossFunction(self):
        return self.optim.loss    

class Autoencoder(DefaultAutoencoder):
    """ AutoEncodeur compresseur d'image """
    def __init__(self, net, fLoss, learning_rate=1e-3):
        DefaultAutoencoder.__init__(self, net, fLoss, learning_rate=learning_rate)
    
    def train(self, features, max_iter=50, size_per_batch = 30):
        """ Fonction de train d'un autoencodeur """
        train_loss = [] # sauvegarder les coûts durant chaque epoch

        for epoch in range(max_iter):
            current_loss = 0 # accumule le coût durant une epoch pour les batch
            allbatches = creer_mini_batches(features, np.ones((features.shape[0], 1)), size_per_batch) # création de batch
            for datax, _ in allbatches:
                self.optim.step(datax, datax)
                current_loss += self.optim.loss_value
            train_loss.append(current_loss/len(allbatches))
            if (epoch+1)%10==0:
                print("Epoch {}/{} avec un coût de {:.3f}".format(epoch+1, max_iter, current_loss/len(allbatches)))
        return train_loss

    def getAutoencoder(self):
        return self.optim.net

    def getLossFunction(self):
        return self.optim.loss

    def reconstruct(self, data):

        output = self.optim.net.forward(data)
        return output


class AutoencoderNoisy(DefaultAutoencoder):
    """ AutoEncodeur débruiteur """
    def __init__(self, net, fLoss, learning_rate=1e-3, noise_eps=0.3):
        DefaultAutoencoder.__init__(self, net, fLoss, learning_rate=learning_rate)
        self.noise_eps = noise_eps
        
    
    def train(self, features, max_iter=50, size_per_batch=30):
        """ Fonction de train d'un autoencodeur """
        train_loss = [] # sauvegarder les coûts durant chaque epoch

        for epoch in range(max_iter):
            current_loss = 0 # accumule le coût durant une epoch pour les batch
            allbatches = creer_mini_batches(features, np.ones((features.shape[0], 1)), size_per_batch) # création de batch
            for datax, _ in allbatches:
                # ajout du bruit aux données
                datax_noisy = add_noise(datax, eps=self.noise_eps)
                #ramener les valeurs entre 0. et 1. par seuillage
                datax_noisy = np.clip(datax_noisy, 0., 1.)
                self.optim.step(datax_noisy, datax)

                current_loss += self.optim.loss_value
            train_loss.append(current_loss/len(allbatches))
            if (epoch+1)%10==0:
                print("Epoch {}/{} avec un coût de {:.3f}".format(epoch+1, max_iter, current_loss/len(allbatches)))
        return train_loss

    def getAutoencoder(self):
        return self.optim.net

    def getLossFunction(self):
        return self.optim.loss

    def reconstruct(self, data):
        """ Reconstruis les données après les avoir bruité. On utilise le réseau appris pour reconstruire les données 
        RETURN: (data_noisy, data_reconstructed)
        """
        data_noisy = add_noise(data, eps=self.noise_eps)
        data_noisy = np.clip(data_noisy, 0.0, 1.0)
        output = self.optim.net.forward(data_noisy)
        return data_noisy, output
    
def add_noise(data, eps = 0.5):
    np.random.seed(1)
    return data + eps * np.random.randn(data.shape[0], data.shape[1])
