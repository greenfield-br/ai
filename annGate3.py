# Fixed neural network for MNIST classification
import numpy
import sys
from include.data import get_mnist
from time import time
from numpy import array, zeros, random, argmax, concatenate, transpose
numpy.set_printoptions(suppress=True, linewidth=1000)
from scipy.special import softmax

class ANN:
    def __init__(self, _in, _out, _KN):
        retCode = 1
        if not isinstance(_in, int) or not isinstance(_out, int) or not isinstance(_KN, list) or not all(isinstance(i, int) for i in _KN):
            retCode = 0
        arrKN = numpy.array(_KN)
        if arrKN.ndim != 1 or arrKN.shape[0] < 1 or any(_e < 1 for _e in arrKN):
            retCode = 0
        if retCode != 1:
            return retCode
        _KN.append(_out)
        self.KN = _KN
        lstANN = []
        Nin = _in
        for counter1 in range(len(_KN)):
            _coefK = zeros((_KN[counter1], Nin+1))
            _coefK[:, :-1] = random.randn(_KN[counter1], Nin) * (2 / Nin)**0.5
            _coefK[:, -1] = zeros(_KN[counter1])
            lstANN.append(_coefK)
            Nin = _KN[counter1]
        self.lst = lstANN

    def f(self, _z, _layer_type='hidden'):
        if _layer_type == 'output':
            return softmax(_z)
        return numpy.maximum(0, _z)

    def Y(self, _IN):
        arrIN = numpy.array(_IN)
        if arrIN.ndim != 1 or arrIN.shape[0] < 1:
            print("Error: Invalid input shape", file=sys.stderr)
            return 0
        lstY = []
        lstZ = []
        lenKN = len(self.KN)
        Kin = arrIN
        for counter1 in range(lenKN):
            _coefK = self.lst[counter1]
            _z = _coefK[:, :-1] @ Kin + _coefK[:, -1]
            _y = self.f(_z, _layer_type='output' if counter1 == lenKN-1 else 'hidden')
            lstZ.append(_z)
            lstY.append(_y)
            Kin = _y
        return lstZ, lstY

def gradJ(_ann, _k, _X, _hatY):
    n = _ann.KN
    lenKN = len(n)
    if _k > lenKN:
        _k = lenKN
    lstZ, lstY = _ann.Y(_X)
    lyrX = _X if lenKN <= 1 else lstY[-2]
    
    if _k == 1:
        delta = lstY[-1] - _hatY  # Shape: (10,)
        weight_grad = numpy.outer(delta, lyrX)  # Shape: (10, 256)
        return numpy.concatenate([weight_grad, delta[:, None]], axis=1)  # Shape: (10, 257)
    
    prev_delta = gradJ(_ann, _k-1, _X, _hatY)[:, -1]  # Shape: (10,)
    weights = _ann.lst[-(_k)][:, :-1]  # Shape: (256, 784)
    z = weights @ _X + _ann.lst[-(_k)][:, -1]  # Shape: (256,)
    delta = numpy.dot(prev_delta, _ann.lst[-(_k-1)][:, :-1]) * (z > 0).astype(float)  # Shape: (256,)
    weight_grad = numpy.outer(delta, _X)  # Shape: (256, 784)
    return numpy.concatenate([weight_grad, delta[:, None]], axis=1)  # Shape: (256, 785)

def iterBackprop(_ann, _lrnCoef, _lstX, _lstHatY, batch_size=32):
    n = _ann.KN
    lenKN = len(n)
    lenLstX = len(_lstX)
    lenLstY = len(_lstHatY)
    minLen = min(lenLstX, lenLstY)
    _lstX = _lstX[:minLen]
    _lstHatY = _lstHatY[:minLen]
    
    for i in range(0, lenLstX, batch_size):
        batch_X = _lstX[i:i+batch_size]
        batch_Y = _lstHatY[i:i+batch_size]
        #print(f"Processing batch {i//batch_size + 1}/{lenLstX//batch_size + 1}", flush=True)
        for layer in range(1, lenKN+1):
            dJ = numpy.zeros_like(_ann.lst[-layer])
            for x, y in zip(batch_X, batch_Y):
                dJ += gradJ(_ann, layer, x, y)
            dJ /= len(batch_X)
            _ann.lst[-layer] -= _lrnCoef * dJ

def cross_entropy_loss(y_pred, y_true):
    return -numpy.sum(y_true * numpy.log(y_pred + 1e-15))

def backprop(_ann, _lrnCoef, _lstX, _lstHatY, _n):
    lenLstX = len(_lstX)
    for count1 in range(_n):
        J = 0
        nr_correct = 0
        iterBackprop(_ann, _lrnCoef, _lstX, _lstHatY)
        for count2 in range(lenLstX):
            z, y = _ann.Y(_lstX[count2])
            y_pred = numpy.array(y[-1])
            J += cross_entropy_loss(y_pred, _lstHatY[count2])
            nr_correct += int(argmax(y_pred) == argmax(_lstHatY[count2]))
        J /= lenLstX
        nr_correct = f"{round((nr_correct / lenLstX) * 100, 2)}%"
        print(f"Epoch {count1}: Loss = {J:.4f}, Accuracy = {nr_correct}, Epochs left = {_n-count1-1}", flush=True)
    return

def saveANN(_ann):
    _t = round(time())
    _filename = f'./data/c{_t}'
    numpy.save(_filename, numpy.array(_ann.lst, dtype=object), allow_pickle=True)

try:
    x = ANN(784, 10, [256])
    images, labels = get_mnist()
    print(f"Loaded {len(images)} training images and {len(labels)} labels", flush=True)
    backprop(x, 0.10, images[:10000], labels[:10000], 5)
    saveANN(x)
except FileNotFoundError as e:
    print(f"Error: Could not load mnist.npz: {e}", file=sys.stderr)
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
