#XOR gate
from numpy import ones, zeros, array, dot, ndarray, eye


class AN:

	def __init__(self, _N):
		self.N = _N				#number of input entries
		self.A = 0.5 * ones(_N)	#array of zeros as default weights
		self.u = 0.5			#bias = 0 as default

	def f(self, _arg=None, _rate=1, _zone=[0, 1], target='y'):					#target = dJ, means you're on a backprop. output is f'(z)
		_z = 0																	#		 = y , means you are evaluating current output 
		if isinstance(_arg, int):     _z = _arg									#either z is given as parameter
		if isinstance(_arg, float):   _z = _arg							
		if isinstance(_arg, list):    _z = dot(self.A, array(_arg)) + self.u	#or it is calculated through _arg
		if isinstance(_arg, ndarray): _z = dot(self.A, _arg) + self.u
		if _z > _zone[0]:
			y = _rate * (_z - _zone[0])
			if target == 'dJ': y = _rate
		if _zone[1] is None: return y
		if _z > _zone[1]:
			y = 2 * _rate * (_zone[1] - _zone[0]) - (_z - _zone[0])
			if target == 'dJ': y = -_rate
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
		lenKN = len(self.KN)
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

	def lstGet(self, target = 'A'): #valid for class properties, not methods.
		_lst = []
		lenKN = len(self.KN)
		for counter1 in range(lenKN):
			_lst.append([])
			lenK = len(self.lst[counter1])
			for counter2 in range(lenK):
				_obj = self.lst[counter1][counter2]
				_val = getattr(_obj, target)
				_lst[counter1].append(_val)
		return _lst

def gradJ(_ann, _X, _hatY):
	n = _ann.KN
	lenKN = len(n)
	lstGradJ = [[]] * lenKN

	arrLyr = zeros(n[-1])
	arrCnt = eye(n[-1])
	lstY = _ann.Y(_X)
	lyrX = _X	#vdefault value in case this is the first layer.
	if lenKN > 1: lyrX = lstY[-2]
	for count in range(n[-1]):
		arrLyr[count] = 2 * (lstY[-1][count] - _hatY[count])
		arrCnt[count][count] = _ann.lst[-1][count].f(lyrX, target='dJ')
	lstGradJ[0] = dot(arrLyr, arrCnt)
	
	for count1 in range(1, lenKN+1):
		if lenKN > count1:
			arrLyr = zeros([n[-count1], n[-(count1+1)]])
			arrCnt = eye(n[-(count1+1)])
			lyrX = _X
			if lenKN > (count1+1): lyrX = lstY[-(count1+2)]
			for count2 in range(n[-count1]):
				arrLyr[count2] = _ann.lst[-count1][count2].A
			for count2 in range(n[-(count1+1)]):
				arrCnt[count2][count2] = _ann.lst[-(count1+1)][count2].f(lyrX, target='dJ')
			lstGradJ[count1] = dot(arrLyr, arrCnt)

	dJ = lstGradJ[0]
	lenLstGradJ = len(lstGradJ)
	for count in range(1, lenLstGradJ):
		print(count, lenLstGradJ)
		dJ = dot(dJ, lstGradJ[count])
	return dJ

def backprop(_ann, _lrnCoef, _iterN, _mode = 'silent'):
	for counter in range(_iterN):
		dJ, J = iterBackprop(_ann, _lrnCoef, [[1, 1], [1, 0], [0, 1], [0, 0]], [[0, 1], [1, 1], [1, 1], [0, 0]])
		if _mode == 'verbose':
			print(counter, dJ, J)
	return


x = ANN(AN(2), 2, 2, [3, 3])
y = gradJ(x, [1, 1], [0, 1])
print(y)
