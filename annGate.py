#XOR gate
from numpy import ones, zeros, array, dot


class AN:

	def __init__(self, _N, _mode = 1):
		self.N = _N				#number of input entries
		self.A = 0.5 * ones(_N)	#array of zeros as default weights
		self.u = 0.5			#bias = 0 as default
		self.t = zeros(2)		#trigger thresholds activate & deactivate
		self.mode = _mode		#activation function behaviour

	def Y(self, _X):
		self.z = dot(self.A, array(_X)) + self.u
		y = 0
		if self.z > 0:
			y = self.z
		if self.mode == 1:
			if self.z > 1:
				y = 2 - self.z
			if self.z > 2:
				y = 0
		return y

	def setA(self, _A):
		arrA = array(_A)
		retCode = 0
		if (arrA.ndim == 1) and (arrA.shape == self.A.shape):
			self.A = arrA
			retCode = 1
		return retCode

	def setu(self, _u):
		retCode = 0
		if isinstance(_u, int) or isinstance(_u, float):
			self.u = _u
			retCode = 1
		return retCode


class ANN:

	def __init__(self, _an, _in, _out, _KN):
		# KN dimension is the number of layers - 1.
		# the last layer is defined by previous and the number of output signals
		retCode = 1 #assume failure by default
		#elementary tests
		if not isinstance(_an, AN):
			retCode = 0
		if not isinstance(_in, int):
			retCode = 0
		if not isinstance(_out, int):
			retCode = 0
		if not isinstance(_KN, list):
			retCode = 0
		if retCode != 1: #exit if failed, as expected.
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
		#ANN building as list of lists of AN class instances
		lenKN = len(arrKN)
		lstANN = []
		Nin = _in #first layer neurons inputs number equals the number of input signals
		for counter1 in range(lenKN):
			lstANN.append([])
			for foo in range(arrKN[counter1]):
				lstANN[-1].append(AN(Nin))
			Nin = arrKN[counter1]
		#adding last layer list. Its neurons number equals the number of output signals
		lstANN.append([])
		for foo in range(_out):
			lstANN[-1].append(AN(Nin))
		#list is complete.
		self.lst = lstANN

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
				


def gradJ(_an, _X, _hatY):
	#initialize neuron
	e = _an.Y(_X) - _hatY
	dJ = zeros(_an.N + 1)
	for counter in range(_an.N):
		dJ[counter] = 0 #assumes activation function is a ramp, not an identity function.
		if _an.z > 0:
			dJ[counter] =  2 * e * 1 * _X[counter]
		if _an.mode == 1: #assumes activation function is a tri, not a ramp.
			if _an.z > 1:
				dJ[counter] = 2 * e * (-1) * _X[counter]
			if _an.z > 2:
				dJ[counter] = 0
	dJ[-1] = 0
	if _an.z > 0:
		dJ[-1] =  2 * e * 1
	if _an.mode == 1:
		if _an.z > 1:
			dJ[-1] = 2 * e * (-1)
		if _an.z > 2:
			dJ[-1] = 0
	return dJ, dot(e, e)


def iterBackprop(_an, _lrnCoef, _lstX, _lstHatY):
	_A = _an.A
	_u = _an.u
	dJ = zeros(_an.N + 1)
	J = 0
	for counter in range(len(_lstX)):
		dj, j = gradJ(_an, array(_lstX[counter]), array(_lstHatY[counter]))
		dJ += dj
		J  += j
	dJ /= len(_lstX)
	_A -= _lrnCoef * dJ[:-1]
	_u -= _lrnCoef * dJ[-1]
	foo = _an.setA(_A)
	foo = _an.setu(_u)
	return dJ, J


def backprop(_an, _lrnCoef, _iterN, _mode = 'silent'):
	for counter in range(_iterN):
		dJ, J = iterBackprop(_an, _lrnCoef, [[1, 1], [1, 0], [0, 1], [0, 0]], [0, 1, 1, 0])
		if _mode == 'verbose':
			print(counter, dJ, _an.A, _an.u, J)
	return


"""
L0 = [AN(2, 0), AN(2, 0)]
L1 = [AN(2, 0)]
_lrnCoef = 0.1
X0 = [1, 1]
X1 = [L0[0].Y(X0), L0[1].Y(X0)]
print(L1[0].A, L1[0].u)
dJ, J = iterBackprop(L1[0], _lrnCoef, [X1], [0])
print(L1[0].A, L1[0].u)
"""

x = ANN(AN(2,0), 2, 2, [2, 3, 4, 5])
#print(x.an.u)
print(x.lst)
