import os

with open(os.path.abspath(__file__).replace("util.py","./setting.py"), "r") as f:
	code = f.read()
	exec(code)



def gumbel_sampling(shape):

	l = int(xp.prod(xp.array(shape)))
	gn = xp.random.rand(l).reshape(shape).astype(xp.float32)
	gn = xp.clip(gn,0.001,0.999)

	gn = chainer.Variable(gn)
	gn = -F.log(-F.log(gn))

	return gn



def gumbel_softmax(p, temp):

	shape = p.shape
	gn = gumbel_sampling(shape)

	z = p + 0.001
	z = F.log(z)
	z += gn

	z = F.softmax(z/temp)
	return z


def multi_samp(p):
	
	shape = p.data.shape
	z = gumbel_softmax(p, temp=1)
	
	order = xp.argsort(z.data)
	
	inx0 = xp.arange(shape[0])
	inx1 = order[:,-2]
	inx2 = order[:,-1]

	z = xp.zeros(shape)
	z[inx0,inx1] = 1
	z[inx0,inx2] = 1

	z = chainer.Variable(z.astype(xp.float32))
	return z


def reinforce_grad(a, param, reward):

	p = F.softmax(param)
	grad = reward * (a-p).data
	return grad


"""
def adv_grad(a, param, eadv):

	p = F.softmax(param)

	wadv_inx = cp.argmax((eadv)[:,:7], axis=1)
	wadv = eadv[cp.arange(14),wadv_inx]


	z.grad[:,7] = wadv
	z.grad[cp.where(z.data==0)] = 0




	inx = cp.argmax(z.data,axis=1)
	adv = z.grad[cp.arange(14),inx]
	adv = adv.reshape(14,1)


	#eadv = np.mean(z.grad[cp.where(z.data!=0)])

	#adv[cp.where(inx==7)] = total_loss

	param.grad = adv * (z-p).data

	return z.grad
"""


def adv_grad(a, param, ea, zero_pos, with_b=False):

	p = F.softmax(param)
	
	ag = a.grad.copy()


	if not (zero_pos is None):
		raw_grad = a.grad.copy()
		eadv = ea.get()
		mask = 1*(a.data == 0)
		eadv = raw_grad + mask * eadv
		eadv = ea(eadv)
		worst = xp.max(eadv, axis=1)
		ag[:,zero_pos] = worst

	if with_b:
		raw_grad = a.grad.copy()
		grad = xp.sum(a.data * raw_grad, axis=1)
		ag = grad - ea.get()
		ea(grad)
		ag2 = xp.zeros(a.data.shape).astype(xp.float32)
		for i in range(a.data.shape[0]):
			ag2[i,:] = a.data[i] * ag[i]

		#print(ag2)
		ag = ag2
	adv = xp.sum(a.data * ag, axis=1)
	adv = adv.reshape((p.shape[0],1))

	grad = adv * (a-p).data

	return grad



"""
def gdas_grad(a, param):

	grad = (param.grad * a).data
	return grad
"""

# (use as reference) https://gist.github.com/robintibor/83064d708cdcb311e4b453a28b8dfdca
# but we use this https://github.com/mit-han-lab/proxylessnas/blob/master/search/modules/mix_op.py


def proxyless_grad(a, param):

	npos = param.data.shape[0]
	ncand = param.data.shape[1]

	inxs = xp.where(a.data!=0)
	
	p = F.softmax(param[inxs].reshape((npos,2))).data
	advs = a.grad[inxs].reshape((npos,2))
	inxs = inxs[1].reshape((npos,2))

	grad = xp.zeros((npos,ncand)).astype(xp.float32)

	for i in range(2):
		inx_i = inxs[:,i]
		for j in range(2):
			grad[xp.arange(npos),inx_i] += advs[:,j] * p[:,j] * (1.*(i==j) - p[:,i])

	return grad


def rescale(oarc, narc, mask):

	npos = oarc.shape[0]
	ncand = oarc.shape[1]

	inx = xp.where(mask!=0)[1].reshape((npos,2))
	inx1 = inx[:,0]
	inx2 = inx[:,1]

	ratio = xp.sum(xp.exp(narc), axis=1) / xp.sum(xp.exp(oarc), axis=1)
	offset = xp.log(ratio + 0.0001)


	narc[:,inx1] -= offset
	narc[:,inx2] -= offset

	return narc





def escape():
	#newp[xp.where(mask!=0)] *= ratio.repeat(2)


	#p_r_0 = newp[xp.where(mask!=0)].reshape((npos,2))[:,0]
	#p_r_1 = newp[xp.where(mask!=0)].reshape((npos,2))[:,1]

	p_r_0 = ratio * newp[:,inx1]
	p_r_1 = ratio * newp[:,inx2]
	
	new_sum_a = xp.sum(xp.exp(narc[xp.where(mask==0)]).reshape((npos,ncand-2)),axis=1)


	eps = 0.0000
	"""
	new_a_0 = cp.log((new_sum_a * (p_r_0 + ((p_r_0 * p_r_1)/(1-p_r_1)))) / (
        1 - p_r_0 - ((p_r_0*p_r_1) / (1-p_r_1))))
	new_a_1 = cp.log((new_sum_a * (p_r_1 + ((p_r_1 * p_r_0)/(1-p_r_0)))) / (
        1 - p_r_1 - ((p_r_1*p_r_0) / (1-p_r_0))))
    """

	new_a_0 = xp.log(eps + (new_sum_a * (p_r_0 + ((p_r_0 * p_r_1)/(1-p_r_1)))) / (eps +
        1 - p_r_0 - ((p_r_0*p_r_1) / (1-p_r_1))))
	new_a_1 = xp.log(eps + (new_sum_a * (p_r_1 + ((p_r_1 * p_r_0)/(1-p_r_0)))) / (eps +
        1 - p_r_1 - ((p_r_1*p_r_0) / (1-p_r_0))))


	narc[:,inx1] = new_a_0
	narc[:,inx2] = new_a_1
	
	return narc


def nasp_c2_const(param):

	x = param.data.copy()
	vmin = xp.min(x, axis=1)
	vmin *= (vmin<0)
	vmin = vmin.reshape(param.shape[0],1)
	x -= vmin

	vmax = xp.max(x, axis=1)
	vmax -= 1.
	vmax *= (vmax>0)
	vmax += 1.
	vmax = vmax.reshape(param.shape[0],1)
	x /= vmax

	return x

def binomial_sampling(n, p):
	return int(np.random.binomial(n,p,1))
"""
def k2a(k, n):
	a = xp.zeros((1,n))
	a[0,int(k)] = 1
	a = a.astype(xp.float32)
	a = chainer.Variable(a)
	return a
"""

def binomial_grad(a, param, n, reward=None):
	p = F.sigmoid(param)
	k = xp.argmax(a.data, axis=1)
	if reward is None:
		adv = a.grad[xp.arange(a.shape[0]),k]
	else:
		adv = xp.ones(param.shape) * reward
	
	k = k.reshape((param.shape))
	adv = adv.reshape(param.shape)

	#lpok = F.log((math.factorial(n)/(math.factorial(k)*math.factorial(n-k))))
	#lpok += F.log(p) * k
	#lpok += F.log(1-p) * (n-k)


	gp = 0.
	gp += k / p
	gp += (n-k) / (1-p) * -1
	gp *= adv


	gp *= (1-p)*p

	return gp.data.reshape(param.shape)




def set_binomial_prior(param, n):

	print("Set Prior")
	print(n)
	for k in range(n+1):
		s = 0.5
		p = (math.factorial(n)/(math.factorial(k)*math.factorial(n-k)))
		p *= s**k
		p *= (1-s) ** (n-k)
		param[:,k] = math.log(p)

	return param


"""
# develop
def adv_grad2(a, param, ea):

	p = F.softmax(param)

	eai = ea(a.data.copy())
	delta = a.data - eai

	adv = xp.sum(delta * a.grad, axis=1)
	adv = adv.reshape((p.shape[0],1))
	grad = adv * (a-p).data

	return grad


# develop
def adv_grad3(a, param, ea):

	p = F.softmax(param)

	adv = xp.sum(a.data * a.grad, axis=1)
	eadv = ea(adv.copy())
	adv -= eadv

	adv = adv.reshape((p.shape[0],1))
	grad = adv * (a-p).data

	return grad

"""


def adv_grad2(a, param):

	p = F.softmax(param)
	
	ag = a.grad.copy()


	a2 = a.data.copy()
	a2 = a2 - a2 * p.data
	adv = xp.sum(a2 * ag, axis=1)
	adv = adv.reshape((p.shape[0],1))

	grad = adv * (a-p).data

	return grad




class Param_Family(chainer.ChainList):
	def __init__(self, mode, shape, max_iter, options={'zero_pos':None, 'prior':'uniform'}, mode2='CAT', n_params=1):
		super(Param_Family, self).__init__()
		with self.init_scope():

			self.n_params = n_params
			for i in range(self.n_params):
				self.add_link(Param(mode=mode, shape=shape, max_iter=max_iter, options=options, mode2=mode2))


			# CAUTION ONRY FOR dir"PROXY"
			params = self.children()
			for i in range(self.n_params):
				normal = params.__next__()
				normal.optimizer.beta1 = 0.5
				normal.optimizer.add_hook(chainer.optimizer.WeightDecay(0.003))
				normal.optimizer.add_hook(chainer.optimizer.GradientClipping(1.0))	


	def set_alpha(self, alpha):

		params = self.children()
		for i in range(self.n_params):
			normal = params.__next__()
			normal.optimizer.alpha = alpha


	def deter(self):

		params = self.children()
		res = []

		for i in range(self.n_params):
			param = params.__next__()
			res.append(param.deter())

		return res

	def draw(self):

		params = self.children()
		res = []

		for i in range(self.n_params):
			param = params.__next__()
			res.append(param.draw())

		return res

	def update(self, argv=None):

		params = self.children()
		res = []

		for i in range(self.n_params):
			param = params.__next__()
			res.append(param.update(argv))

		return res




class Param(chainer.Chain):
	def __init__(self, mode, shape, max_iter, options={'zero_pos':None, 'prior':'uniform'}, mode2='CAT'):
		super(Param, self).__init__()
		with self.init_scope():
			self.param = chainer.Parameter((xp.ones(shape)*1.).astype(xp.float32))
			self.param.data[:,0] += 0.000001

			self.npos = shape[0]
			self.ncand = shape[1]
			
			self.N = self.ncand - 1
			self.mode2 = mode2
			if self.mode2 == 'BI':
			    self.param = chainer.Parameter(xp.zeros((self.npos,1)).astype(xp.float32))

			elif 'prior' in options and options['prior'] == 'unimodal':
			    self.param.data = set_binomial_prior(self.param.data, self.N)

			self.mode = mode
			self.max_iter = max_iter
			
			self.optimizer = chainer.optimizers.Adam(alpha=0.001,beta1=0.5)
			#self.optimizer = chainer.optimizers.SGD(lr=0.001)
			#self.optimizer = chainer.optimizers.SGD(lr=0.001)
			#self.optimizer = chainer.optimizers.Adam(alpha=0.001)
			self.optimizer.setup(self)
			#self.optimizer.add_hook(chainer.optimizer.WeightDecay(0.003))
			#self.optimizer.add_hook(chainer.optimizer.GradientClipping(1.0))

			#### FROM 2019 code
			#optimizer2 = chainer.optimizers.Adam(beta1=0.5, beta2=0.999)
			#optimizer2.add_hook(chainer.optimizer.WeightDecay(0.003))

			self.options = options

			if mode == "A" or mode == "A2":
				self.ea = EMA()

			if mode == "S" or mode == "GD":
				self.n_iter = 0

			if mode == "PA":
				self.rewards = []
				self.a_save = []


			#self.grad_mean = EMA()




			
	def deter(self):
		if self.mode2 != "BI":
			return F.softmax(self.param/ZERO, axis=1)
		else:
			a = xp.zeros((self.npos, self.ncand)).astype(xp.float32)
			for i in range(self.npos):
				p = float(F.sigmoid(self.param[i]).data)
				mode = xp.floor((self.N+1.)*p)
				a[i,int(mode)] = 1

			return chainer.Variable(a)


	def draw(self, nasp_random=False):
		
		self.cleargrads()

		if self.mode == "R" or self.mode == "A" or self.mode == "PA":
			p = F.softmax(self.param, axis=1)
			self.a = gumbel_softmax(p, temp=ZERO)

		# develop			
		if self.mode == "A2" or self.mode == "A3":
			p = F.softmax(self.param, axis=1)
			self.a = gumbel_softmax(p, temp=ZERO)



		if self.mode == "D":
			self.a = F.softmax(self.param)

		if self.mode == "S":
			lrcoef = (math.cos(self.n_iter*math.pi/self.max_iter)+1.0)/2.0
			p = F.softmax(self.param, axis=1)
			self.a = gumbel_softmax(p, temp=lrcoef)

		if self.mode == "GD":
			lrcoef = 10 - 9.9*(self.n_iter/self.max_iter)
			p = F.softmax(self.param, axis=1)
			z = gumbel_softmax(p, temp=lrcoef)

			targ = F.softmax(z/ZERO)
			diff = targ.data - z.data
			self.a = z + diff
			
		if self.mode == "P":
			p = F.softmax(self.param, axis=1)
			self.a = multi_samp(p)

		if self.mode == "N":
			self.param.data = nasp_c2_const(self.param)
			if nasp_random:
				self.a = gumbel_softmax(F.softmax(self.param, axis=1), temp=ZERO)
			else:
				self.a = F.softmax(self.param/ZERO, axis=1)

		if self.mode2 == "BI":
			a = xp.zeros((self.npos, self.ncand)).astype(xp.float32)
			for i in range(self.npos):
				p = float(F.sigmoid(self.param[i]).data)
				k = binomial_sampling(self.N, p)
				a[i,k] = 1

			self.a = chainer.Variable(a)
			

			
		return self.a



	def update(self, argv=None):

		if self.mode == "R" and self.mode2 == "CAT":
			reward = argv[0]
			self.param.grad = reinforce_grad(self.a, self.param, reward)

		if self.mode == "A" and self.mode2 == "CAT":
			self.param.grad = adv_grad(self.a, self.param, self.ea, zero_pos=self.options['zero_pos'])
			
		if self.mode == "A2":
			self.param.grad = adv_grad(self.a, self.param, self.ea, zero_pos=self.options['zero_pos'], with_b=True)
			
		#if self.mode == "A3":
		#	self.param.grad = adv_grad3(self.a, self.param, self.ea)

		if self.mode == "P":
			old_param = self.param.data.copy()
			self.param.grad = proxyless_grad(self.a, self.param)

		if self.mode == "S" or self.mode == "GD":
			self.n_iter += 1

		if self.mode == "PA":
			reward = argv[0]
			self.rewards.append(reward)
			self.a_save.append(self.a)

			if len(self.rewards) == 8:
				w = xp.array(self.rewards).reshape((1,8))
				w = F.softmax(w).data[0]
				self.param.grad = xp.zeros(self.param.shape).astype(xp.float32)
				for i in range(8):
				    if self.mode2 == "CAT":
				        self.param.grad += reinforce_grad(self.a_save[i], self.param, float(w[i]))
				    else:
				        self.param.grad += binomial_grad(self.a_save[i], self.param, self.N, reward=float(w[i]))


				self.rewards = []
				self.a_save = []

		#if self.mode == "GD":
		#	self.param.grad = gdas_grad(self.a, self.param)
		
		if self.mode == "N":
			self.param.grad = self.a.grad

		if self.mode2 == "BI":
		    if self.mode == "R":
		        reward = argv[0]
		        self.param.grad = binomial_grad(self.a, self.param, self.N, reward=reward)
		    if self.mode == "A":
		        self.param.grad = binomial_grad(self.a, self.param, self.N, reward=None)

		param_bef = self.param.data.copy()
		self.optimizer.update()


		if self.mode == "P":
			self.param.data = rescale(old_param, self.param.data, self.a.data)

		if self.mode == "N":
			self.param.data = nasp_c2_const(self.param)

	
		#grad_var(param_bef, param.data)


			

		

class EMA():
	def __init__(self, coef=0.05):
		self.y = None
		self.coef = coef

	def __call__(self, y):
		if self.y is None:
			self.y = y
		else:
			self.y = self.coef*y + (1.0-self.coef)*self.y

		return self.y

	def get(self):
		if self.y is None:
			return 0.
		else:
			return self.y











from chainer.dataset import convert

from chainer import training
from chainer.training import extension
from chainer.training import extensions






import math
#import chainercv.transforms.image as cv
import time
import os
import pickle
import random
import sys


#from basemodel import *
#import cifar

















def cutout(img, csize):

	h = img.shape[1]
	w = img.shape[2]

	y = np.random.randint(h)
	x = np.random.randint(w)

	y1 = np.clip(y - csize // 2, 0, h)
	y2 = np.clip(y + csize // 2, 0, h)
	x1 = np.clip(x - csize // 2, 0, w)
	x2 = np.clip(x + csize // 2, 0, w)

	img[ : , y1:y2 , x1:x2] = 0.0

	return img



class Preprocess(chainer.dataset.DatasetMixin):
	def __init__(self, pairs, with_cutout, test):
		self.pairs = pairs
		self.with_cutout = with_cutout
		self.test = test

	def __len__(self):
		return len(self.pairs)

	def get_example(self, i):
		x, y = self.pairs[i]
		x = x.copy()

		# label
		y = np.array(y, dtype=np.int32)

		if self.test:
			return x, y

		# random crop
		pad_x = np.zeros((3, 40, 40), dtype=np.float32)
		pad_x[:, 4:36, 4:36] = x
		top = random.randint(0, 8)
		left = random.randint(0, 8)
		x = pad_x[:, top:top + 32, left:left + 32]
		# horizontal flip
		if random.randint(0, 1):
			x = x[:, :, ::-1]

		if self.with_cutout:
			x = cutout(x,16)

		#plt.imshow(x.transpose((1,2,0))+0.5)
		#plt.show()

		return x, y




class CosineSchedule(extension.Extension):

	"""Trainer extension to exponentially shift an optimizer attribute.
    This extension exponentially increases or decreases the specified attribute
    of the optimizer. The typical use case is an exponential decay of the
    learning rate.
    This extension is also called before the training loop starts by default.
    Args:
        attr (str): Name of the attribute to shift.
        rate (float): Rate of the exponential shift. This value is multiplied
            to the attribute at each call.
        init (float): Initial value of the attribute. If it is ``None``, the
            extension extracts the attribute at the first call and uses it as
            the initial value.
        optimizer (~chainer.Optimizer): Target optimizer to adjust the
            attribute. If it is ``None``, the main optimizer of the updater is
            used.
	"""

	def __init__(self, attr, max_epoch, train_iter, optimizer=None):
		self._attr = attr
		self._max_epoch = max_epoch
		self._train_iter = train_iter
		self._init = None
		self._optimizer = optimizer
		self._last_value = None

	def initialize(self, trainer):
		optimizer = self._get_optimizer(trainer)
		# ensure that _init is set
		if self._init is None:
			self._init = getattr(optimizer, self._attr)


	def __call__(self, trainer):

		optimizer = self._get_optimizer(trainer)

		epoch = self._train_iter.epoch
		lrcoef = (math.cos(epoch * math.pi / self._max_epoch) + 1.0) / 2.0
		value = self._init * lrcoef
		self._update_value(optimizer, value)


	"""
    def serialize(self, serializer):
        self._t = serializer('_t', self._t)
        self._last_value = serializer('_last_value', self._last_value)
        if isinstance(self._last_value, numpy.ndarray):
            self._last_value = self._last_value.item()
	"""
	def _get_optimizer(self, trainer):
		return self._optimizer or trainer.updater.get_optimizer('main')

	def _update_value(self, optimizer, value):
		setattr(optimizer, self._attr, value)
		self._last_value = value

		




def plot_batch(batch):
	import matplotlib.pyplot as plt
	from PIL import Image
	batch = cuda.to_cpu(batch)

	batch -= np.min(batch)
	batch /= np.max(batch)
	batch *= 255

	dst = Image.new('RGB', (36*4, 36*16), (0,0,0))
	
	for x in range(4):
		for y in range(16):	
			inx = 16*x + y
			img = batch[inx].astype(np.uint8)
			img = img.transpose((1,2,0))
			img = Image.fromarray(img)
			dst.paste(img, (36*x, 36*y))


	dst.save("./batch.png")
	exit()