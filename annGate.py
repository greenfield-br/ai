#XOR gate
from numpy import ones, zeros, array, dot, ndarray


class AN:

	def __init__(self, _N):
		self.N = _N				#number of input entries
		self.A = 0.5 * ones(_N)	#array of zeros as default weights
		self.u = 0.5			#bias = 0 as default

	def f(self, _arg=None, _rate=1, _zone=[0, 1], _target='y'):					#_target = dJ, means you're on a backprop. output is f'(z)
		_z = 0																	#		 = y , means you are evaluating current output 
		if isinstance(_arg, int):     _z = _arg									#either z is given as parameter
		if isinstance(_arg, float):   _z = _arg							
		if isinstance(_arg, list):    _z = dot(self.A, array(_arg)) + self.u	#or it is calculated through _arg
		if isinstance(_arg, ndarray): _z = dot(self.A, _arg) + self.u
		y = 0
		if _z > _zone[0]:
			y = _rate * (_z - _zone[0])
			if _target == 'dJ': y = _rate
		if _zone[1] is None: return y
		if _z > _zone[1]:
			y = 2 * _rate * (_zone[1] - _zone[0]) - (_z - _zone[0])
			if _target == 'dJ': y = -_rate
		if _z > 2 * _rate * (_zone[1] - _zone[0]) + _zone[0]:
			y = 0
		return y

	def Y(self, _X):
		if isinstance(_X, list):    _z = dot(self.A, array(_X)) + self.u
		if isinstance(_X, ndarray): _z = dot(self.A, _X) + self.u
		y = self.f(_z)
		return y


class ANN:

	def __init__(self, _an, _in, _out, _KN):
		# KN dimension is the number of hidden layers (total - 1).
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
			lstANN.append([])
			for foo in range(self.KN[counter1]):
				lstANN[-1].append(AN(Nin))
			Nin = self.KN[counter1]
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
		lstY = []
		lenKN = len(self.lst)
		Kin = _IN
		for counter1 in range(lenKN):
			lstY.append([])
			lenK = len(self.lst[counter1])
			for counter2 in range(lenK):
				_obj = self.lst[counter1][counter2]
				_val = getattr(_obj, 'Y')(Kin)
				lstY[counter1].append(_val)
			Kin = lstY[counter1]
		return lstY

	def lstGet(self, _target = 'A'): #valid for class properties, not methods.
		_lst = []
		lenKN = len(self.lst)
		for counter1 in range(lenKN):
			_lst.append([])
			lenK = len(self.lst[counter1])
			for counter2 in range(lenK):
				_obj = self.lst[counter1][counter2]
				_val = getattr(_obj, _target)
				_lst[counter1].append(_val)
		return _lst


def gradJ(_ann, _X, _hatY):
	hatY = array(_hatY)
	lstY = _ann.Y(_X)		#feedforward input
	lyrX = array(lstY[-2])	#input values are outputs of previous layer
	lyrY = array(lstY[-1])	#output is current layer neuron output list
	e = lyrY - hatY
	lenK = len(lstY)
	Nin = len(lyrX)
	Nk = len(lyrY)
	dJ = zeros([Nk, Nin+1])
	for counter1 in range(Nk):
		for counter2 in range(Nin):
			dJ[counter1][counter2]  = _ann.lst[-1][counter1].f(lyrX, _target = 'dJ')	#f() derivative
			dJ[counter1][counter2] *= 2 * e[counter1] * lyrX[counter2]					#dJ module
		dJ[counter1][-1]  = _ann.lst[-1][counter1].f(lyrX, _target = 'dJ')
		dJ[counter1][-1] *= 2 * e[counter1] * 1
	return dJ, dot(e, e)

def iterBackprop(_ann, _lrnCoef, _lstX, _lstHatY):
	lenK = len(_ann.lst[-1])						#number of neurons on output layer
	_A = [_ann.lst[-1][i].A for i in range(lenK)]
	_u = [_ann.lst[-1][i].u for i in range(lenK)]
	dJ = zeros([lenK, _ann.lst[-1][0].N+1])
	J = 0
	lenLstX = len(_lstX)
	for counter in range(lenLstX):
		[dj, j] = gradJ2(_ann, _lstX[counter], _lstHatY[counter])
		dJ += dj
		J  += j
	dJ /= lenLstX
	_A -= _lrnCoef * array([i[:-1] for i in dJ])
	_u -= _lrnCoef * array([i[ -1] for i in dJ])
	for counter in range(lenK):
		w = _ann.lst[-1][counter]
		setattr(w, 'A', _A[counter])
		setattr(w, 'u', _u[counter])
	return dJ, J

def backprop(_ann, _lrnCoef, _iterN, _mode = 'silent'):
	for counter in range(_iterN):
		dJ, J = iterBackprop(_ann, _lrnCoef, [[1, 1], [1, 0], [0, 1], [0, 0]], [[0, 1], [1, 1], [1, 1], [0, 0]])
		if _mode == 'verbose':
			print(counter, dJ, J)
	return



#x = ANN(AN(2), 2, 2, [3, 3])

#
# need to rewrite gradJ according to matricial structure.
#

def gradJ2(_ann, _X, _hatY, _k=1):
	hatY = array(_hatY)
	lstY = _ann.Y(_X)		#feedforward input
	lyrX = array(lstY[-2])	#input values are outputs of previous layer
	lyrY = array(lstY[-1])	#output is current layer neuron output list
	e = lyrY - hatY
	lenK = len(lstY)
	Nin = len(lyrX)
	Nk = len(lyrY)
	dJ = zeros([Nk, Nin+1])
	
	#rewrite this for loop where lyrX[i] is the split point for parallelization, thus
	#rewrite it for looping over all the layer neurons under the same input parameter, and then
	#               looping over all the layer neurons under other    input parameter.
	for counter1 in range(Nin):
		for counter2 in range(Nk):
			dJ[counter2][counter1]  = _ann.lst[-1][counter2].f(lyrX, _target='dJ')	#f() derivative
			dJ[counter2][counter1] *= 2 * e[counter2] * lyrX[counter1]					#dJ module
	for counter in range(Nk):
		dJ[counter][-1]  = _ann.lst[-1][counter].f(lyrX, _target='dJ')
		dJ[counter][-1] *= 2 * e[counter] * 1
	return dJ, dot(e, e)

x = ANN(AN(2), 2, 2, [3, 3])
y = x.Y([1, 1])
print(y)

a, b = iterBackprop(x, 0.1, [[1, 1]], [[0, 1]])
print(a)
print(b)
y = x.Y([1, 1])
print(y)

a, b = iterBackprop(x, 0.1, [[1, 1]], [[0, 1]])
print(a)
print(b)
y = x.Y([1, 1])
print(y)

#now i need to check again against google sheet.
