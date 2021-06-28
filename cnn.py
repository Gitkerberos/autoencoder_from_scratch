from conv import Conv1D
from relu import Relu
from maxpool import MaxPool1D
from flatten import Flatten
from sequentiel import Sequentiel
from module import Linear
import numpy as np
from loss import LogSoftMax
from optim import Optim
import matplotlib.pyplot as plt
from softmax import SoftMax
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

def get_usps(l,datax,datay):
    if type(l)!=list:
        resx = datax[datay==l,:]
        resy = datay[datay==l]
        return resx,resy
    tmp =   list(zip(*[get_usps(i,datax,datay) for i in l]))
    tmpx,tmpy = np.vstack(tmp[0]),np.hstack(tmp[1])
    return tmpx,tmpy

def show_usps(data):
    plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")



def test_cnn():
    
    cnn = Sequentiel([
        Conv1D(4, 1, 3, 4),
        MaxPool1D(2, 2),
        Flatten(),
        Linear(96, 50),
        Relu(),
        Linear(50, 10)
    ])

    o = Optim(cnn, LogSoftMax(), 0.001)

    uspsdatatrain = "USPS_train.txt"
    uspsdatatest = "USPS_test.txt"
    alltrainx, alltrainy = load_usps(uspsdatatrain)
    alltestx, alltesty = load_usps(uspsdatatest)
    classes = [i for i in range(10)]
    samples_training = [get_usps([c], alltrainx, alltrainy) for c in classes]
    samples_test = [get_usps([c], alltestx, alltesty) for c in classes]
    classes_one_hot = np.array([np.zeros(10) for i in range(10)])
    for i in range(10):
        classes_one_hot[i,i] = 1
    sizes_training = [ (len(x),len(y)) for (x,y) in samples_training ]

    datax = np.array([x for (tx,ty) in samples_training for x in tx])
    datay_raw = np.array([y for (tex,tey) in samples_training for y in tey])

    o.step(datax,datay_raw)

    testx = np.array([x for (tx,ty) in samples_test for x in tx])
    # bruit
    #testx = testx + 0.4 * np.random.randn(testx.shape[0], testx.shape[1])
    testy_raw = np.array([y for (tex,tey) in samples_test for y in tey])

    print("===================== TESTING ========================")

    out = cnn.forward(testx)
    s = SoftMax()
    out_class = s.forward(out)
    print("\n\n\n\n")

    score = 0
    total = 0

    for c in range(len(out_class)):
        total += 1
        if np.argmax(out_class[c])==np.argmax(testy_raw[c]):
            score += 1
    print("score : ", score,"/",total, " = ", score/total)

    outp = np.argmax(out_class, axis=1)
    #print(testy_raw[])
    print(np.unique(outp))
    # matrice de confusion
    
    figure = plt.figure()
    axes = figure.add_subplot(111)
    
    # using the matshow() function 
    caxes = axes.matshow(confusion_matrix(testy_raw, outp), interpolation ='nearest')
    figure.colorbar(caxes)

    plt.savefig("../matrice_conf_CNN_2.jpg")
    plt.show()



def test_small_cnn():
    cnn = Sequentiel([
        Conv1D(2, 1, 3, 2),
        MaxPool1D(2, 2),
        Flatten(),
        Linear(27, 10),
        Relu(),
        Linear(10, 3)
    ])

    o = Optim(cnn, LogSoftMax())

    datax = np.array([[0,0,0,1,0,0,0,0,1,1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0],
                      [0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,1,0,1,0,0,1,0,0,1,0,0,0,0,0,1,0,0],
                      [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,0, 1, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,0, 1, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,0, 1, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],

                      [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                       0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
                       0, 1, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                       0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
                       0, 1, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                       0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
                       0, 1, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                       0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
                       0, 1, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                       0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
                       0, 1, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                       0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
                       0, 1, 0, 0],
    ])

    o.step(datax,
           np.array([[1,0,0], [1,0,0], [1,0,0], [1,0,0], [1,0,0], [1,0,0], [1,0,0], [1,0,0], [1,0,0], [1,0,0], [1,0,0], [1,0,0],

                     [1,0,0], [1,0,0], [1,0,0], [1,0,0], [1,0,0], [1,0,0], [1,0,0], [1,0,0], [1,0,0], [1,0,0], [1,0,0], [1,0,0]]))
    out = cnn.forward(datax)
    s = SoftMax()
    out_class = s.forward(out)
    print("\n\n\n\n", out_class)


if __name__ =="__main__":
    test_cnn()
