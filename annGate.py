#XOR gate
from numpy import ones, zeros, array, dot


class ANN:

	def __init__(self, _N, _mode = 1):
		self.N = _N				#number of input entries
		self.X = zeros(_N)		#array of zeros as default input
		self.A = 0.5 * ones(_N)	#array of zeros as default weights
		self.u = 0.5			#bias = 0 as default
		self.t = zeros(2)		#trigger thresholds activate & deactivate
		self.mode = _mode		#activation function behaviour

	def Y(self):
		self.z = dot(self.A, self.X) + self.u
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
		self.A = _A

	def setu(self, _u):
		self.u = _u


class Layer:

	def __init__(self, _N, _an, _anN, _anMode = 1):
		self.N  = _N	#number of ANN in the layer
		self.an = _an
		self.anN = _anN
		self.anMode = _anMode
		self.lstANN = []
		for counter in range(self.N):
			self.lstANN.append(self.an(self.anN, self.anMode))


def gradJ(_an, _X, _hatY):
	#initialize neuron
	_an.X = _X
	e = _an.Y() - _hatY
	dJ = zeros(_an.N+1)
	for counter in range(_an.N):
		dJ[counter] = 0 #assumes activation function is a ramp, not an identity function.
		if _an.z > 0:
			dJ[counter] =  2 * e * 1 * _an.X[counter]
		if _an.mode == 1:
			if _an.z > 1:
				dJ[counter] = 2 * e * (-1) * _an.X[counter]
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


def iterBackprop(_an, _lstX, _lstHatY):
	dJ = zeros(_an.N+1)
	J = 0
	for counter in range(len(_lstX)):
		dj, j = gradJ(_an, _lstX[counter], _lstHatY[counter])
		dJ += dj
		J  += j
	dJ /= len(_lstX)
	return dJ, J


def backprop(_an, _lrnCoef, _iterN, _mode = 'silent'):
	for counter in range(_iterN):
		dJ, J = iterBackprop(_an, [[1, 1], [1, 0], [0, 1], [0, 0]], [0, 1, 1, 0])
		_an.A -= _lrnCoef * dJ[:-1]
		_an.u -= _lrnCoef * dJ[-1]
		if _mode == 'verbose':
			print(counter, dJ, _an.A, _an.u, J)
	return


#ann = [ANN(2), ANN(3)]
#print(ann[1].u)

#L = Layer(1, ANN, 2)
#backprop(L.lstANN[0], 0.5, 20, 'verbose')

L0 = Layer(2, ANN, 2)
L1 = Layer(1, ANN, 2)

_lrnCoef = 0.5
for counter2 in range(5):
	for counter in range(len(L1.lstANN)):
		_A = L1.lstANN[counter].A
		_u = L1.lstANN[counter].u
		dJ, J = iterBackprop(L1.lstANN[counter], [[1, 1], [1, 0], [0, 1], [0, 0]], [0, 1, 1, 0])
		_A -= _lrnCoef * dJ[:-1]
		_u -= _lrnCoef * dJ[-1]
		L1.lstANN[counter].setA = _A
		L1.lstANN[counter].setu = _u
		print(L1.lstANN[counter].A, L1.lstANN[counter].u)
