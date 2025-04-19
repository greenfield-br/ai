# Fixed neural network for MNIST classification
from numpy import zeros, ones, array, ndarray, eye, random, argmax, save, transpose
from include.data import get_mnist
from time import time
from numpy import set_printoptions
set_printoptions(suppress=True, linewidth=1000)
from scipy.special import softmax

class ANN:
    def __init__(self, _in, _out, _KN):
        # Initialize network with input size, output size, and hidden layer sizes
        retCode = 1
        if not isinstance(_in, int) or not isinstance(_out, int) or not isinstance(_KN, list) or not all(isinstance(i, int) for i in _KN):
            retCode = 0
        arrKN = array(_KN)
        if arrKN.ndim != 1 or arrKN.shape[0] < 1 or any(_e < 1 for _e in arrKN):
            retCode = 0
        if retCode != 1:
            return retCode
        _KN.append(_out)
        self.KN = _KN
        # Build weight matrices with Xavier initialization
        lstANN = []
        Nin = _in
        for counter1 in range(len(_KN)):
            _coefK = zeros((_KN[counter1], Nin+1))
            _coefK[:, :-1] = random.randn(_KN[counter1], Nin) * (2 / Nin)**0.5  # Xavier init
            _coefK[:, -1] = zeros(_KN[counter1])  # Zero biases
            lstANN.append(_coefK)
            Nin = _KN[counter1]
        self.lst = lstANN

    def f(self, _z, _layer_type='hidden'):
        # Activation: ReLU for hidden layers, Softmax for output
        if _layer_type == 'output':
            return softmax(_z)
        return np.maximum(0, _z)  # ReLU

    def Y(self, _IN):
        # Forward pass with ReLU/Softmax
        arrIN = array(_IN)
        if arrIN.ndim != 1 or arrIN.shape[0] < 1:
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
            Kin = lstY[counter1]
        return lstZ, lstY

def gradJ(_ann, _k, _X, _hatY):
    # Compute gradients for backpropagation
    n = _ann.KN
    lenKN = len(n)
    if _k > lenKN:
        _k = lenKN
    lstZ, lstY = _ann.Y(_X)
    lyrX = _X if lenKN <= 1 else lstY[-2]
    
    # Output layer (Softmax + cross-entropy)
    if _k == 1:
        delta = lstY[-1] - _hatY  # Gradient of cross-entropy with Softmax
        return np.concatenate([delta @ lyrX[None, :], delta])  # Weights and bias
    
    # Hidden layers (ReLU)
    prev_delta = gradJ(_ann, _k-1, _X, _hatY)[:n[-(_k-1)]]
    weights = _ann.lst[-(_k-1)][:, :-1]
    z = weights @ lyrX + _ann.lst[-(_k-1)][:, -1]
    delta = prev_delta @ weights * (z > 0).astype(float)  # ReLU derivative
    return np.concatenate([delta @ lyrX[None, :], delta])

def iterBackprop(_ann, _lrnCoef, _lstX, _lstHatY, batch_size=32):
    # Backpropagation with mini-batches
    n = _ann.KN
    lenKN = len(n)
    lenLstX = len(_lstX)
    lenLstY = len(_lstHatY)
    minLen = min(lenLstX, lenLstY)
    _lstX = _lstX[:minLen]
    _lstHatY = _lstHatY[:minLen]
    
    # Process mini-batches
    for i in range(0, lenLstX, batch_size):
        batch_X = _lstX[i:i+batch_size]
        batch_Y = _lstHatY[i:i+batch_size]
        for layer in range(1, lenKN+1):
            dJ = np.zeros_like(_ann.lst[-layer])
            for x, y in zip(batch_X, batch_Y):
                dJ += gradJ(_ann, layer, x, y)
            dJ /= len(batch_X)
            _ann.lst[-layer][:, :-1] -= _lrnCoef * dJ[:-1]
            _ann.lst[-layer][:, -1] -= _lrnCoef * dJ[-1]

def cross_entropy_loss(y_pred, y_true):
    # Cross-entropy loss
    return -np.sum(y_true * np.log(y_pred + 1e-15))

def backprop(_ann, _lrnCoef, _lstX, _lstHatY, _n):
    # Training loop with cross-entropy loss
    lenLstX = len(_lstX)
    for count1 in range(_n):
        J = 0
        nr_correct = 0
        iterBackprop(_ann, _lrnCoef, _lstX, _lstHatY)
        for count2 in range(lenLstX):
            z, y = _ann.Y(_lstX[count2])
            y_pred = np.array(y[-1])
            J += cross_entropy_loss(y_pred, _lstHatY[count2])
            nr_correct += int(argmax(y_pred) == argmax(_lstHatY[count2]))
        J /= lenLstX
        nr_correct = f"{round((nr_correct / lenLstX) * 100, 2)}%"
        print(f"Epoch {count1}: Loss = {J:.4f}, Accuracy = {nr_correct}, Epochs left = {_n-count1-1}")
    return

def saveANN(_ann):
    # Save weights
    _t = round(time())
    _filename = f'./data/c{_t}'
    save(_filename, array(_ann.lst, dtype=object), allow_pickle=True)

# Initialize and train network
x = ANN(784, 10, [256])  # Larger hidden layer
images, labels = get_mnist()  # Already normalized
backprop(x, 0.10, images[:10000], labels[:10000], 50)  # More data, fewer epochs
saveANN(x)
