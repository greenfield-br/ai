#XOR gate
from numpy import zeros, ones, array, ndarray, eye, multiply, random, argmax, exp, diag, transpose, save, load, reshape
from include.data import get_mnist
from time import time
from numpy import set_printoptions
set_printoptions(suppress=True, linewidth=1000)
from scipy.special import softmax

class ANN:

	def __init__(self, _in, _out, _KN):
		# KN dimension is the number of hidden layers (total - 1).
		# the last layer is defined by previous and the number of output signals
		retCode = 1 #assume failure by default
		#elementary tests
		if not isinstance(_in, int):
			retCode = 0
		if not isinstance(_out, int):
			retCode = 0
		if not isinstance(_KN, list):
			retCode = 0
		if not all(isinstance(i, int) for i in _KN):
			retCode = 0
		if retCode != 1:	#exit if failed, as expected.
			return retCode
		#array structure tests
		arrKN = array(_KN)
		if arrKN.ndim != 1:
			retCode = 0
		if arrKN.shape[0] < 1:
			retCode = 0
		if retCode != 1:
			return retCode
		#array content tests
		if any(_e < 1 for _e in arrKN):
			retCode = 0
		if retCode != 1:
			return retCode
		_KN.append(_out)	#_KN is now extended to the total number of layers
		self.KN = _KN		#it does not need to be an array
		#ANN building as list of lists of AN class instances
		lstANN = []
		lenKN = len(_KN)
		#first layer neurons inputs number equals the number of input signals
		Nin = _in
		for counter1 in range(lenKN):
			_coefK = zeros((_KN[counter1], Nin+1))
			_coefK[:, :-1] = random.rand(_KN[counter1], Nin) - 0.5 * ones((_KN[counter1], Nin))
			_coefK[:, -1] = zeros(_KN[counter1])
			lstANN.append(_coefK)
			Nin = self.KN[counter1]
		self.lst = lstANN

	def f(self, _z, _rate=1, _zone=[0, 1]):
		y = zeros(_z.shape)
		_fArg = (_z > _zone[0]).astype(float)
		y = _rate * (_z - _zone[0]) @ diag(_fArg)

		if _zone[1] is None: return y

		_fArg  = (_z >= _zone[1]).astype(float)
		_fNArg = (_z <  _zone[1]).astype(float)
		y = y @ diag(_fNArg)
		y += (2 * _rate * _zone[1] - _rate * (_z - _zone[0])) @ diag(_fArg)

		_fArg  = (_z <= 2 * _zone[1]).astype(float)
		y = y @ diag(_fArg)
		return y

	def Y(self, _IN):
		retCode = 1
		#array structure tests
		arrIN = array(_IN)
		if arrIN.ndim != 1:
			retCode = 0
		if arrIN.shape[0] < 1:
			retCode = 0
		if retCode != 1:
			return retCode
		lstY = []
		lstZ = []
		lenKN = len(self.KN)
		Kin = arrIN
		for counter1 in range(lenKN):
			_coefK = self.lst[counter1]
			_z = _coefK[:, :-1] @ Kin + _coefK[:, -1]
			_y = self.f(_z)
			lstZ.append(_z)
			lstY.append(_y)
			Kin = lstY[counter1]
		return lstZ, lstY


def gradJ(_ann, _k, _X, _hatY):
	n = _ann.KN
	lenKN = len(n)
	if _k > lenKN: _k = lenKN
	lstGradJ = [[]] * _k

	lstZ, lstY = _ann.Y(_X)
	
	lyrX = _X						#default value in case this is the first layer.
	if lenKN > 1: lyrX = lstY[-2]	#if there is more than 1 layer, input is the previous layer output.

	y = lstY[-1]
	y = softmax(y)
	arrLyr = 2 * (y - _hatY)

	lenK = n[-1]
	_df = lstY[-1] >= lstZ[-1]
	_df = (_df.astype(float) - 0.5 * ones(lenK)) * 2

	arrCnt = diag(_df)

	lstGradJ[0] = arrLyr @ arrCnt

	#in case dJ is ran on a layer before the last one. assumes it ran already on the last one.
	if _k > 1:
		for count1 in range(1, _k):
			lyrX = _X
			if lenKN > (count1+1): lyrX = lstY[-(count1+2)]

			_coefK = _ann.lst[-count1][:, :-1]
			arrLyr = _coefK

			lenK = n[-(count1+1)]
			_df = lstY[-(count1+1)] >= lstZ[-(count1+1)]
			_df = (_df.astype(float) - 0.5 * ones(lenK)) * 2
			arrCnt = diag(_df)

			lstGradJ[count1] = arrLyr @ arrCnt

	#build proper lstGradJ expression by multiplying each layer component.
	lenLstGradJ = len(lstGradJ)
	dj = lstGradJ[0]
	for count in range(1, lenLstGradJ):
		dj = dj @ lstGradJ[count]

	#replicates lstGradJ by each input signal element to build dJ vector
	arrDj = array(dj)
	arrLyrX = array(list(lyrX) + [1])
	arrDj.shape   += (1,)
	arrLyrX.shape += (1,)
	a = arrDj @ transpose(arrLyrX)
	dJ = transpose(a)

	return dJ

def iterBackprop(_ann, _lrnCoef, _lstX, _lstHatY):
	n = _ann.KN
	lenKN = len(n)
	lenLstX = len(_lstX)
	lenLstY = len(_lstHatY)
	minLen = min(lenLstX, lenLstY)
	_lstX = _lstX[:minLen]
	_lstHatY = _lstHatY[:minLen]
	lenLstX = len(_lstX)

	#averages dJ over all input output pairs at k-th layer
	for count1 in range(1, lenKN+1):

		dJ = gradJ(_ann, count1, _lstX[0], _lstHatY[0])
		for count2 in range(1, lenLstX):
			dJ += gradJ(_ann, count1, _lstX[count2], _lstHatY[count2])
		dJ /= lenLstX

		_coefK = _ann.lst[-count1]
		_coefK[:, :-1] -= _lrnCoef * transpose(dJ[:-1])
		_coefK[:, -1]  -= _lrnCoef * transpose(dJ[-1])
		_lst = _ann.lst
		_lst[-count1] = _coefK
		setattr(_ann, 'lst', _lst)
	return

def backprop(_ann, _lrnCoef, _lstX, _lstHatY, _n):
	lenLstX = len(_lstX)
	for count1 in range(_n):
		J = 0
		nr_correct = 0
		iterBackprop(_ann, _lrnCoef, _lstX, _lstHatY)
		for count2 in range(lenLstX):
			z, y = _ann.Y(_lstX[count2])
			e = array(y[-1]) - array(_lstHatY[count2])
			J += e @ e
			J /= lenLstX
			nr_correct += int(argmax(y[-1]) == argmax(_lstHatY[count2]))

		nr_correct = f"{round((nr_correct / lenLstX) * 100, 2)}%"
		print(J, nr_correct, _n-count1)
		#print(_ann.lst)

#	for count in range(lenLstX):
#		z, y = _ann.Y(_lstX[count])
#		print(y[-1], _lstHatY[count])
	
	return


def saveANN(_ann):
	_t = round(time())
	_filename = 'c' + str(_t)
	_filename = './data/' + _filename
	save(_filename, array(_ann.lst, dtype=object), allow_pickle=True)




x = ANN(784, 10, [20])

images, labels = get_mnist()
images /= 255

backprop(x, 0.10, images[:1000], labels[:1000], 100)

saveANN(x)

#_lst = load('./data/c1717988024.npy', allow_pickle=True)
#print(_lst)
