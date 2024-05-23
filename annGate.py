#XOR gate
from numpy import ones, zeros, array, dot, ndarray


class AN:

	def __init__(self, _N):
		self.N = _N				#number of input entries
		self.A = 0.5 * ones(_N)	#array of zeros as default weights
		self.u = 0.5			#bias = 0 as default

	def f(self, _arg=None, _rate=1, _zone=[0, 1], _target='y'):
		_z = 0
		if isinstance(_arg, int):   _z = _arg								#either z is given as parameter
		if isinstance(_arg, float): _z = _arg								
		if isinstance(_arg, list):  _z = dot(self.A, array(_arg)) + self.u	#or it is calculated through _arg
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
		z = dot(self.A, array(_X)) + self.u
		y = self.f(z)
		return y


class ANN:

	def __init__(self, _an, _in, _out, _KN):
		#1 KN dimension is the number of layers - 1.
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
		self.arrKN = array(_KN)
		if self.arrKN.ndim != 1:
			retCode = 0
		if self.arrKN.shape[0] < 1:
			retCode = 0
		if retCode != 1:
			return retCode
		#array content tests
		if any(_e < 1 for _e in self.arrKN):
			retCode = 0
		if retCode != 1:
			return retCode
		#ANN building as list of lists of AN class instances
		lstANN = []
		lenKN = len(self.arrKN)
		#first layer neurons inputs number equals the number of input signals
		Nin = _in
		for counter1 in range(lenKN):
			lstANN.append([])
			for foo in range(self.arrKN[counter1]):
				lstANN[-1].append(AN(Nin))
			Nin = self.arrKN[counter1]
		#adding last layer list. Its neurons number equals the number of output signals
		lstANN.append([])
		for foo in range(_out):
			lstANN[-1].append(AN(Nin))
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

	def lstGet(self, _target = 'A'):
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


def gradJ(_lst, _X, _hatY):
	hatY = array(_hatY)
	lstY = _lst.Y(_X)
	lyrX = array(lstY[-2])
	lyrY = array(lstY[-1])
	e = lyrY - hatY
	lenK = len(lstY)
	Nin = len(lyrX)
	Nk = len(lyrY)
	dJ = zeros([Nk, Nin+1])
	for counter1 in range(Nk):
		for counter2 in range(Nin):
			dJ[counter1][counter2]  = _lst.lst[-1][counter1].f(_X, _target = 'dJ')	#f() derivative
			dJ[counter1][counter2] *= 2 * e[counter1] * lyrX[counter2]				#dJ module
		dJ[counter1][-1]  = _lst.lst[-1][counter1].f(_X, _target = 'dJ')
		dJ[counter1][-1] *= 2 * e[counter1] * 1
	return dJ, dot(e, e)

def iterBackprop(_lst, _lrnCoef, _lstX, _lstHatY):
	lenK = len(_lst.lst[-1])
	_A = [_lst.lst[-1][i].A for i in range(lenK)]
	_u = [_lst.lst[-1][i].u for i in range(lenK)]
	dJ = zeros([lenK, _lst.lst[-1][0].N+1])
	J = 0
	for counter in range(len(_lstX)):
		[dj, j] = gradJ(_lst, _lstX[counter], _lstHatY[counter])
		dJ += dj
		J  += j
	dJ /= len(_lstX)
	_A -= _lrnCoef * array([i[:-1] for i in dJ])
	_u -= _lrnCoef * array([i[ -1] for i in dJ])
	for counter in range(lenK):
		w = _lst.lst[-1][counter]
		setattr(w, 'A', _A[counter])
		setattr(w, 'u', _u[counter])
	_A = [_lst.lst[-1][i].A for i in range(lenK)]
	return dJ, J

def backprop(_an, _lrnCoef, _iterN, _mode = 'silent'):
	for counter in range(_iterN):
		dJ, J = iterBackprop(_an, _lrnCoef, [[1, 1], [1, 0], [0, 1], [0, 0]], [0, 1, 1, 0])
		if _mode == 'verbose':
			print(counter, dJ, J)
	return


x = AN(2)
y = x.Y(0.3)
print(y)
