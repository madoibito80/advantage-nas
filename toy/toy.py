# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import pickle

import time
import os
import math

import sys
sys.path.append("../")

#from util import *
import util


with open(os.path.abspath(__file__).replace("toy.py","../setting.py"), "r") as f:
	code = f.read()
	exec(code)



scale = 1.#/10000.

ntrial = 10
niter = 2000

#modes = ["A","D","N"]
#modes = ["N","GD","A","P"]
#modes = ["GD","A","BI"]#, "P"]

mode_labels = {"BI":"Binomial","N":"NASP","PA":"PARSEC","GD":"GDAS", "A":"proposed method", "P":"ProxylessNAS", "R":"REINFORCE", "D":"DARTS","S":"SNAS", "W":"rand arc&train weight"}
line_styles = {"BI":"solid","PA":"solid","GD":"solid", "A":"dashed", "P":"dashdot", "R":"dotted", "D":"solid","S":"solid", "W":"solid"}
orders = {"BI":1,"N":0,"PA":1,"GD":1,"A":5,"R":2,"S":4,"D":3,"P":0, "W":6}

colors = {"BI":"cyan","N":"gray","PA":"pink", "A":'#1f77b4', "P":'#ff7f0e', "D":'#2ca02c', "S":'#d62728', "R":'#9467bd', "GD":'#8c564b', "A2":'#1f77b4'}



BINOMIAL = False


modes = ["R","N","GD","D","PA","S","P","A"]
modes2 = ["CAT","CAT","CAT","CAT","CAT","CAT","CAT","CAT"]

#modes = ["A"]
#modes2 = ["CAT"]


#modes = ["GD","A2","A","R"]
#modes2 = ["CAT","CAT", "CAT", "CAT"]


batch_size = 100

npos = 10
ncand = 10

xdim = 1

W = 13
H = 13


def forward(x, a, operations, second, mode):

	for pos in range(npos):
		operations[pos].cleargrads()
		for op in second[pos]:
			op.cleargrads()


	y = 0.
	for pos in range(npos):
		h1 = operations[pos](x)
		h2 = calc_op(a, h1, pos, mode)
		h2 = F.tanh(h2)

		h = h2
		for op in second[pos]:
			h = op(h)
		y += h

	y *= 1./npos
	#y = F.sigmoid(y)
	return y



def calc_op(a, x, pos, mode):

	onehotter = ["BI","GD","PA","A","P","R","A2"]
	h = 0.
	for cand in range(ncand):
		if mode in onehotter and float(a[pos,cand].data) == 0.:
			skip = True
		else:
			c = x.data.copy()
			if BINOMIAL:
				c[:,cand+1:] = 0.
				c = xp.sum(c, axis=1)
				#c /= (cand+1.)
			else:
				c = c[:,cand]
				#if cand == 0:
				#	c[:,0] *= 0.

			c = c.reshape((batch_size,1,4,4))

			h += F.scale(c, a[pos,cand])
			
		if mode == "GD" and float(a[pos,cand].data) == 0.:
			z = 1. + (0. * x.data[:,0].reshape((batch_size,1,4,4)))
			h += F.scale(z, a[pos,cand])

	return h






	
def batch_sample(batch_size):
	x = xp.random.rand(batch_size*W*H).astype(xp.float32).reshape((batch_size,1,W,H))
	x *= 2.
	x -= 1.
	x = chainer.Variable(x)
	return x








def softmax_cross_entropy(x, t):
	y = F.softmax(x)
	y = F.log(y)

	tp = xp.zeros((batch_size,2))
	tp[xp.arange(batch_size),t.data] = 1

	loss = -F.sum(tp * y)
	return loss


def toy_lossfunc(y,t):
	#t = F.argmax(t, axis=1)
	#return softmax_cross_entropy(y, t)
	#return F.sigmoid_cross_entropy(y, t)
	return F.mean_squared_error(y, t)


def main(tryc):

	try:
		os.mkdir("./output/"+sys.argv[2])
	except:
		print("mkdir error")

	try:
		os.mkdir("./output/"+sys.argv[2]+"/"+str(tryc))
	except:
		print("mkdir error")

		
	operations = []
	for i in range(npos):
		layer = L.Convolution2D(1,ncand,ksize=7,stride=2)#,nobias=True)
		layer.to_gpu()
		layer.disable_update()
		#layer.W.data += 0.9
		operations.append(layer)
	
	second = []
	for i in range(npos):
		layer1 = L.Convolution2D(1,1,ksize=4,stride=1,nobias=True)
		#layer0 = L.BatchNormalization(1)
		#layer0.to_gpu()
		#layer0.disable_update()

		#layer1 = L.Linear(16,1)
		layer1.to_gpu()
		layer1.disable_update()
		second.append([layer1])

	a = chainer.Parameter(xp.random.rand(npos*ncand).reshape((npos,ncand)).astype(xp.float32))
	teacher = F.softmax(a/ZERO, axis=1)

	f = open("./output/"+sys.argv[2]+"/"+str(tryc)+"/teacher", "wb")
	pickle.dump(teacher, f)
	pickle.dump(operations, f)
	pickle.dump(second, f)
	f.close()


	for q in range(len(modes)):
		mode = modes[q]
		mode2 = modes2[q]
		start = None

		losses = []
		timer = []
		orders = []

		if BINOMIAL:
			options={'zero_pos':None, 'prior':'unimodal'}
		else:
			options={'zero_pos':None, 'prior':'uniform'}
		
		param = util.Param(mode=mode, shape=(npos, ncand), max_iter=niter, options=options, mode2=mode2)

		##############
		"""
		# set prior
		if mode == "BI":
			ps = []
			for k in range(ncand):
				n = ncand-1
				s = float(F.sigmoid(param.param[0]).data)
				p = (math.factorial(n)/(math.factorial(k)*math.factorial(n-k)))
				p *= s**k
				p *= (1-s) ** (n-k)
				ps.append(p)
		else:
			param.param.data = set_binomial_prior(param.param.data)
		
			ps = F.softmax(param.param, axis=1).data[0]
			ps = cuda.to_cpu(ps).tolist()
			#ps.extend([0,0,0,0,0,0,0,0,0,0])
			
		print(len(ps))
		plt.bar(np.arange(ncand), ps, alpha=0.5)
		plt.show()
		exit()
		"""




		baseline = util.EMA(coef=0.05)
		#acct = EMA(coef=0.2)
		#acct = MA(steps=10)


		for i in range(niter):

			if i % 10 == 0:

				#print(param.param)
				
				if start is None:
					timer.append(0.)
				else:
					timer.append(time.time() - start)

				x = batch_sample(100)
				t = forward(x, teacher, operations, second, mode)
				a = param.deter()
				#order = xp.argmax(a.data)
				#print(order)
				#orders.append(order)
				y = forward(x, a, operations, second, mode)

				#loss = float(F.mean_squared_error(y,t).data) * scale
				loss = float(toy_lossfunc(y,t).data)
				losses.append(loss)
				start = time.time()
				print(tryc, mode, i, losses[-1], timer[-1])
				#print(param.param)


			lrcoef = (math.cos(i*math.pi/niter)+1.0)/2.0
			param.optimizer.alpha = 0.001# * lrcoef
			param.optimizer.beta1 = 0.9

			x = batch_sample(batch_size)
			t = forward(x, teacher, operations, second, mode)
			a = param.draw()
			y = forward(x, a, operations, second, mode)
			#loss = F.mean_squared_error(y,t)	* scale
			loss = toy_lossfunc(y,t)

			loss.backward(retain_grad=True)
			baseline(float(loss.data))
			#hist.append(baseline.get())

			if mode == "R":
				param.update([float(loss.data)-baseline.get()])
			elif mode == "PA": 
				param.update([float(loss.data)])
			else:
				param.update([float(loss.data)-baseline.get()])
				



		f = open("./output/"+sys.argv[2]+"/"+str(tryc)+'/'+mode+"_"+mode2, 'wb')
		pickle.dump(losses, f)
		pickle.dump(timer, f)
		pickle.dump(orders, f)
		pickle.dump(param, f)
		f.close()


	
def trial(ntrial):

	for k in range(ntrial):
		main(tryc=k)





"""
def plot_loss():

	phist = {}
	pacc = {}

	for mode in modes:
		loss = None
		acc = None

		for k in range(ntrial):
			f = open('./output/'+str(k), 'rb')
			hists = pickle.load(f)
			accs = pickle.load(f)

			if loss is None:
				loss = np.array(hists[mode])
				acc = np.array(accs[mode])
			else:
				loss = np.c_[loss,np.array(hists[mode])]
				acc = np.c_[acc,np.array(accs[mode])]

			f.close()
			

		#upp_loss = np.percentile(loss,75,axis=1)
		#mid_loss = np.percentile(loss,50,axis=1)
		#low_loss = np.percentile(loss,25,axis=1)

		try:
			mid_loss = np.mean(loss, axis=1)
			std = np.std(loss, axis=1) * 0.2
			upp_loss = mid_loss + std
			low_loss = mid_loss - std
		except:
			mid_loss = loss
			upp_loss = loss
			low_loss = loss

		phist[mode] = [upp_loss,mid_loss,low_loss]


	#plt.rcParams["font.size"] = 15

	cut = 1.	# how long plot (ratio of maximum iteration)
	
	for mode in modes:
		upp = phist[mode][0][:int(niter*cut)]
		low = phist[mode][2][:int(niter*cut)]
		x = np.linspace(1,int(niter*cut),int(niter*cut))
		plt.fill_between(x,upp,low,where=upp>low,alpha=0.6,label=mode_labels[mode],zorder=orders[mode],color=colors[mode])
		plt.plot(phist[mode][1],color=colors[mode])

	plt.xlabel("iteration")
	plt.ylabel("cross entropy loss")
	plt.legend()
	#plt.ylim(0,1)
	plt.savefig("./output/loss.png")
	plt.cla()
"""



def plot_acc(with_time=False):

	import matplotlib


	import matplotlib
	matplotlib.rcParams['ps.useafm'] = True
	matplotlib.rcParams['pdf.use14corefonts'] = True


	font_size = 9
	plt.rcParams["font.size"] = font_size

	plt.figure(figsize=(3.3,1.6))


	#plt.grid(b=True)


	phist = {}
	ptimer = {}
	pacc = {}

	xmax = 9999
	for q in range(len(modes)):
		mode = modes[q]
		mode2 = modes2[q]

		loss = None
		timer = None
		acc = None

		for k in range(ntrial):
			try:
				f = open('./output/'+sys.argv[2]+"/"+str(k)+'/'+mode+'_'+mode2, 'rb')
			except:
				print('./output/'+sys.argv[2]+"/"+str(k)+'/'+mode+'_'+mode2)
				exit()
				continue
			loss_co = pickle.load(f)
			timer_co = pickle.load(f)
			if loss is None:
				loss = np.array(loss_co)
				timer = np.array(timer_co)
			else:
				loss = np.c_[loss,np.array(loss_co)]
				timer = np.c_[timer,np.array(timer_co)]
				
			f.close()


		loss *= 100.
			

		try:
			mid_loss = np.percentile(loss,50, axis=1)
			upp_loss = np.percentile(loss,75, axis=1)
			low_loss = np.percentile(loss,25, axis=1)
			#std = np.std(loss, axis=1)
			#upp_loss = mid_loss + std
			#low_loss = mid_loss - std
		except:
			if loss is None:
				continue
			mid_loss = loss
			upp_loss = loss
			low_loss = loss

		phist[mode] = [upp_loss,mid_loss,low_loss]
		span = np.mean(timer)
		print(mode, span)
		ptimer[mode] = span


	#plt.rcParams["font.size"] = 15

		cut = 1.	# how long plot (ratio of maximum iteration)
		upp = phist[mode][0][:int(niter*cut)]
		low = phist[mode][2][:int(niter*cut)]
		x = np.linspace(1,int(len(loss)),int(len(loss)))*10
		if with_time:
			x = np.linspace(1,int(len(loss)),int(len(loss)))*ptimer[mode]*10
			if xmax > ptimer[mode]:
				xmax = ptimer[mode]

		#plt.fill_between(x, upp,low,where=upp>low,alpha=0.6, color=COLORS[mode])#,zorder=orders[mode],color=colors[mode])
		#try:
		#	print(hfui)
		#	print(ffnj)
		#	plt.plot(x, phist[mode][1], color=colors[mode], label=mode_labels[mode])
		#except:
		plt.fill_between(x, upp,low,where=upp>low,alpha=0.35, color=COLORS[mode])
		plt.plot(x, phist[mode][1], color=COLORS[mode], label=LAVELS[mode], linewidth=1.)

	#plt.ylim(-0.05, 1.5)
	plt.xlabel("iteration")
	#plt.ylabel("approximation error")
	plt.ylabel("loss")
	#plt.legend(loc='upper left', bbox_to_anchor=(0, 1.5), ncol=2)
	plt.legend(loc='upper left', bbox_to_anchor=(0.98,1.08), ncol=1,handlelength=0.8,handletextpad=0.3,labelspacing=0.29,edgecolor="#ffffff")

	plt.xlim(0,600)

	#plt.ylim(-0.001,0.025)
	#plt.ylim(-10,1000)
	#plt.ylim(-0.001,0.005)
	if with_time:
		plt.xlim(0,xmax*len(phist[modes[0]][1])*10)
		plt.xlabel("time(sec)")
	#plt.subplots_adjust(left=0.2, right=0.8, bottom=0.1, top=0.93)
	plt.subplots_adjust(left=0.105, right=0.67, bottom=0.24, top=0.98)
	plt.savefig("./output/toy.pdf")#, bbox_inches='tight', pad_inches=0.0)
	plt.cla()
	#plt.clf()






if __name__ == "__main__":

	if sys.argv[1] == "t":
		trial(ntrial)
	if sys.argv[1] == "v":
		plot_acc()
	if sys.argv[1] == "vt":
		plot_acc(with_time=True)
