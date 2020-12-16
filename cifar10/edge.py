# -*- coding: utf-8 -*-

import os

with open(os.path.abspath(__file__).replace("edge.py","../setting.py"), "r") as f:
	code = f.read()
	exec(code)

from chainer.dataset import convert
from util import *

"""
def set_conv(stock, in_c, out_c, ksize, groups, stride, dilate):

	param = stock.W.data.copy()
	param = param[:out_c,:in_c,:ksize,:ksize]

	inw = int(in_c/groups)
	outw = int(out_c/groups)

	w = cp.zeros((out_c,inw,ksize,ksize)).astype(cp.float32)
	for cgn in range(int(out_c/outw)):
		w[cgn*outw:(cgn+1)*outw] = param[cgn*outw:(cgn+1)*outw,cgn*inw:(cgn+1)*inw,:,:]


	pad = int((ksize+1)/3)
	if dilate == 2:
		pad *= 2

	b = stock.b.data.copy()
	b = b[:out_c]

	conv_inst = L.Convolution2D(in_c, out_c, ksize=ksize, pad=pad, stride=stride, dilate=dilate, groups=groups)
	conv_inst.to_gpu()
	conv_inst.W.data = w
	conv_inst.b.data = b

	return conv_inst


def set_bn(stock, out_c):

	bn_inst = L.BatchNormalization(out_c, use_gamma=False, use_beta=False)

	if False:

		param = stock.gamma.data.copy()
		param = cuda.to_cpu(param)
		param = param[:out_c]
		bn_inst.gamma.data = param

		param = stock.beta.data.copy()
		param = cuda.to_cpu(param)
		param = param[:out_c]
		bn_inst.beta.data = param

	bn_inst.to_gpu()
	#bn_inst.beta.data = cuda.to_gpu(bn_inst.beta.data)
	#bn_inst.gamma.data = cuda.to_gpu(bn_inst.gamma.data)

	return bn_inst

	# DARTS
	# we always use batch-specific
	# statistics for batch normalization rather than the global moving average. Learnable affine parameters
	# in all batch normalizations are disabled during the search process to avoid rescaling the outputs of the
	# candidate operations.




def put_conv(stock, inst, in_c, out_c, ksize, groups):

	param = inst.W.data.copy()

	inw = int(in_c/groups)
	outw = int(out_c/groups)

	for cgn in range(int(out_c/outw)):
		stock.W.data[cgn*outw:(cgn+1)*outw,cgn*inw:(cgn+1)*inw,:ksize,:ksize] = param[cgn*outw:(cgn+1)*outw]

	param = inst.b.data.copy()
	stock.b.data[:out_c] = param



def put_bn(stock, inst, out_c):

	stock.to_cpu()
	inst.to_cpu()

	stock.gamma.data[:out_c] = inst.gamma.data
	stock.beta.data[:out_c] = inst.beta.data 

	stock.to_gpu()
	inst.to_gpu()
"""




def adjast(x, c):

	if x.data.shape[1] < c:
		pad = chainer.Variable(cp.zeros((x.shape[0],c-x.shape[1],x.shape[2],x.shape[3])).astype(cp.float32))
		x = F.concat((x,pad),axis=1)
	elif x.data.shape[1] > c:
		x = x[:,:c]

	return x





class Edge(chainer.Chain):

	def __init__(self, max_c, stride=1, bg=True):
		super(Edge, self).__init__()
		with self.init_scope():

			bg = False
			self.conv1 = L.Convolution2D(max_c, max_c, ksize=1, pad=0, stride=1, groups=1)
			self.bn1 = L.BatchNormalization(max_c, use_gamma=bg, use_beta=bg)
			self.conv3 = L.Convolution2D(max_c, max_c, ksize=3, pad=1, stride=1, groups=1)
			self.bn3 = L.BatchNormalization(max_c, use_gamma=bg, use_beta=bg)

			self.max_c = max_c
			self.is_eval = False

			# zero opとmax pool周りでアドバンテージがになりがち
			# constantを0から始めたとこと多少マシに




	def __call__(self, x, a, mode, is_nasp_upd_w=False):

		y = 0
		for op_type in range(len(a)):

			if (mode in ONEHOTTER or is_nasp_upd_w) and float(a[op_type].data) == 0.:
				skip = True

			else:

				if op_type == 0:
					h = x * 1.
				
				if op_type == 1:
					h = F.relu(x)
					h = self.conv1(x)
					h = self.bn1(h)
				
				if op_type == 2:
					h = F.relu(x)
					h = self.conv3(x)
					h = self.bn3(h)

				if op_type == 3:
					h = F.average_pooling_2d(x, ksize=3, stride=1, pad=1)

				if op_type == 4:
					h = x * 0.
		
					
				y += F.scale(h, a[op_type])

			if mode == "GD" and float(a[op_type].data) == 0.:
				h = chainer.Variable(xp.ones(x.shape).astype(xp.float32))
				y += F.scale(h, a[op_type])
		

			"""
			if (op_type == 4) and self.stride == 2:
				h = h[:,:,:-1,:-1]

			if (not self.is_eval) and self.mode == "A" and op_type != 7:
				self.zeroc(cp.mean(h.data))

			if not self.is_eval:
				h = F.scale(h, self.mask[op_type])
			y += h

			# path drop
			#if self.is_eval and chainer.config.train and np.random.rand() < 0.2:
			#	y *= 0
			"""
		return y
