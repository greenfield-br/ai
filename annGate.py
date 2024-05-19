#XOR gate
from numpy import ones, zeros, array, dot, ndarray


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
				lstY[counter1].append(self.lst[counter1][counter2].Y(Kin))
			Kin = lstY[counter1]
		return lstY

	def getA(self):
		lstA = []
		lenKN = len(self.lst)
		for counter1 in range(lenKN):
			lstA.append([])
			lenK = len(self.lst[counter1])
			for counter2 in range(lenK):
				lstA[counter1].append(self.lst[counter1][counter2].A)
		return lstA

	def getu(self):
		lstu = []
		lenKN = len(self.lst)
		for counter1 in range(lenKN):
			lstu.append([])
			lenK = len(self.lst[counter1])
			for counter2 in range(lenK):
				lstu[counter1].append(self.lst[counter1][counter2].u)
		return lstu



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
			dJ[counter1][counter2] = 0 #assumes activation function is a ramp, not an identity function.
			if _lst.lst[-1][counter1].z > 0:
				dJ[counter1][counter2] =  2 * e[counter1] * 1 * lyrX[counter2]
			if _lst.lst[-1][counter1].mode == 1: #assumes activation function is a tri, not a ramp.
				if _lst.lst[-1][counter1].z > 1:
					dJ[counter1][counter2] = 2 * e[counter1] * (-1) * lyrX[counter2]
				if _lst.lst[-1][counter1].z > 2:
					dJ[counter1][counter2] = 0
		dJ[counter1][-1] = 0
		if _lst.lst[-1][counter1].z > 0:
			dJ[counter1][-1] =  2 * e[counter1] * 1
		if _lst.lst[-1][counter1].mode == 1:
			if _lst.lst[-1][counter1].z > 1:
				dJ[counter1][-1] = 2 * e[counter1] * (-1)
			if _lst.lst[-1][counter1].z > 2:
				dJ[counter1][-1] = 0
	return dJ, dot(e, e)


def iterBackprop(_lst, _lrnCoef, _lstX, _lstHatY):
	lenK = len(_lst.lst[-1])
	_A = [_lst.lst[-1][i].A for i in range(lenK)] #_lst.lst[-1][0].A
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
		foo = _lst.lst[-1][counter].setA(_A[counter])
		foo = _lst.lst[-1][counter].setu(_u[counter])
	_A = [_lst.lst[-1][i].A for i in range(lenK)] #_lst.lst[-1][0].A
	return dJ, J


def backprop(_an, _lrnCoef, _iterN, _mode = 'silent'):
	for counter in range(_iterN):
		dJ, J = iterBackprop(_an, _lrnCoef, [[1, 1], [1, 0], [0, 1], [0, 0]], [0, 1, 1, 0])
		if _mode == 'verbose':
			print(counter, dJ, J)
	return


x = ANN(AN(2,0), 2, 2, [3, 3])
print(x.getu())
a, b = iterBackprop(x, 0.1, [[1, 1]], [0, 0]) #, [1, 0]], [[0, 1], [0, 0]])
print(x.getu())
a, b = iterBackprop(x, 0.1, [[1, 1]], [0, 0]) #, [1, 0]], [[0, 1], [0, 0]])
#backprop(x, 0.1, 20, 'verbose')
print(x.getu())
