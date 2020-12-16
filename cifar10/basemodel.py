import chainer
import chainer.links as L
import chainer.functions as F
from chainer.dataset import convert
import numpy as np
import cupy as cp

from edge import *





class ResCell(chainer.Chain):

	def __init__(self, in_c, out_c, bg=True):
		super(ResCell, self).__init__()
		with self.init_scope():


			self.conv1 = L.Convolution2D(in_c, out_c, ksize=3, pad=1, stride=2, groups=1)
			self.bn1 = L.BatchNormalization(out_c, use_gamma=bg, use_beta=bg)
			
			self.conv2 = L.Convolution2D(out_c, out_c, ksize=3, pad=1, stride=1, groups=1)
			self.bn2 = L.BatchNormalization(out_c, use_gamma=bg, use_beta=bg)

			self.shortcut = L.Convolution2D(in_c, out_c, ksize=1, pad=0, stride=1, groups=1)


	def __call__(self, x):

		h = F.relu(x)
		h = self.conv1(h)
		h = self.bn1(h)
		h = F.relu(h)
		h = self.conv2(h)
		h1 = self.bn2(h)

		h2 = F.average_pooling_2d(x, ksize=3, stride=2, pad=1)
		h2 = self.shortcut(h2)

		return h1 + h2





class Cell(chainer.ChainList):

	def __init__(self, nc):
		super(Cell, self).__init__()
		with self.init_scope():

			self.c = nc

			for i in range(6):
				edge = Edge(self.c, stride=1, bg=True)
				self.add_link(edge)



	def __call__(self, x, a, mode, is_nasp_upd_w=False):

		node = [x]
		edges = self.children()
		pos = 0
		for i in range(1,4):
			node.append(None)
			for j in range(0,i):
				edge = edges.__next__()
				res = edge(x=node[j], a=a[pos], mode=mode, is_nasp_upd_w=is_nasp_upd_w)
				pos += 1

				if node[i] is None:
					node[i] = res
				else:
					node[i] = F.add(node[i],res)

		return node[3]





class Model(chainer.ChainList):
	def __init__(self):
		super(Model, self).__init__()
		with self.init_scope():

			self.num_stage = 3
			self.num_cells = 5
			self.init_c = 16
			self.c_coef = [1,2,4]
			
			self.add_link(L.Convolution2D(3,self.init_c,ksize=3,pad=1,nobias=True))
			self.add_link(L.BatchNormalization(self.init_c))

			for stage in range(self.num_stage):
				for celln in range(self.num_cells):
					self.add_link(Cell(nc=self.init_c*self.c_coef[stage]))

				if stage+1 != self.num_stage:
					self.add_link(ResCell(in_c=self.init_c*self.c_coef[stage], out_c=self.init_c*self.c_coef[stage+1]))

			
			fc = L.Linear(self.init_c*self.c_coef[-1],10)
			self.add_link(fc)


	def __call__(self, x, a, mode, is_nasp_upd_w=False):

		cells = self.children()

		h = cells.__next__()(x) # init_conv
		h = cells.__next__()(h) # init_bn

		for stage in range(1,self.num_stage+1):
			for celln in range(self.num_cells):
				cell = cells.__next__()
				h = cell(h, a, mode, is_nasp_upd_w)


			if stage != self.num_stage:
				h = cells.__next__()(h)


		h = F.average_pooling_2d(h, h.shape[2:])
		y = cells.__next__()(h) # fully connected

		return y

	"""
	def set_inst(self, normal, reduce, put, mode=None):
		cells = self.children()
		stem = cells.__next__()
		for i in range(self.num_cells):
			cell = cells.__next__()
			if i in self.redpos:
				cell.set_inst(reduce, put, mode)
			else:
				cell.set_inst(normal, put, mode)
	"""





"""
class AuxiliaryHeadCIFAR(chainer.Chain):
	def __init__(self, max_c):
		super(AuxiliaryHeadCIFAR, self).__init__()
		with self.init_scope():
			self.conv1 = L.Convolution2D(max_c*4,128,ksize=1,nobias=True)
			self.bn1 = L.BatchNormalization(128)

			self.conv2 = L.Convolution2D(128,768,ksize=2,nobias=True)
			self.bn2 = L.BatchNormalization(768)

			self.fc = L.Linear(768, 10)

	def __call__(self, x):

		h = F.relu(x) # image size = 8 * 8
		h = F.average_pooling_2d(h, ksize=5,stride=3,pad=0) # image size = 2 * 2

		h = self.conv1(h)
		h = self.bn1(h)
		h = F.relu(h)

		h = self.conv2(h)
		h = self.bn2(h)
		h = F.relu(h)

		h = self.fc(h)
		return h
"""

