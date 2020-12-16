
from chainer.dataset import convert



import math
#import chainercv.transforms.image as cv
import time
import os
import pickle
import random
import sys

import sys
sys.path.append("../")

import util


with open(os.path.abspath(__file__).replace("search.py","../setting.py"), "r") as f:
	code = f.read()
	exec(code)


from basemodel import *
import cifar


gpu = 0
batch_size = 64
max_c = 16
max_epoch = 90

freeze = 50



"""
def regularizer(arc):

	param_size = 0
	cost = cp.array([9+1, 25+1, 9+1, 25+1, 0, 0, 0, 0])

	for i in range(14):
		param = arc[i]
		param_size += F.matmul(param, cost)

	loss = param_size
	return loss
"""


def arc2query(arc):
	# like NAS-Bench-201
	pos = 0
	text = "|"
	for i in range(3):
		for j in range(0,i+1):
			for op_type in range(arc.shape[1]):
				if float(arc[pos,op_type].data) > 0.9:
					if op_type == 0:
						text += "skip_connect"
					if op_type == 1:
						text += "nor_conv_1x1"
					if op_type == 2:
						text += "nor_conv_3x3"
					if op_type == 3:
						text += "avg_pool_3x3"
					if op_type == 4:
						text += "none"
						
					break

			text += "~"
			text += str(j)
			text += "|"
			pos += 1
		if i != 2:
			text += "+|"

	return text


def dump_arc(arc, epoch, start, f, loss=0.):
	# save architecture like NAS-Bench-201
	text = str(epoch) + ' , ' + str(loss) + ' , '
	text += arc2query(arc)
	text += '\r\n'
	f = open('./snapshot/'+str(start)+'/'+f+'_arc.txt','a')
	f.write(text)
	f.close()




def main():
	
	mode = sys.argv[1]

	try:
		resume_pickle = sys.argv[2]
		resume = True
	except:
		resume = False

	print("resume = " + str(resume))


	for i in [0]:
		chainer.backends.cuda.get_device_from_id(i).use()


	model = Model()
	model.to_gpu()

	#if mode == 'A':
	if False:
		options = {'zero_pos':4}
	else:
		options = {'zero_pos':None}

	ncand = 4
		
	#normal = util2.Param(mode=mode, shape=(6,ncand), max_iter=(max_epoch-freeze)*(50000/batch_size), options=options)
	normal = util.Param(mode=mode, shape=(6,ncand), max_iter=(max_epoch-freeze)*(25000/batch_size))
	normal.to_gpu()

	normal.optimizer.beta1 = 0.5
	normal.optimizer.add_hook(chainer.optimizer.WeightDecay(0.003))
	normal.optimizer.add_hook(chainer.optimizer.GradientClipping(1.0))

	
	loss_hist = []
	train_acc = []
	advs = []
	totals = []

	
	(train, val, test) = cifar.get_cifar10()
	
	train = Preprocess(train, with_cutout=False, test=False)
	val = Preprocess(val, with_cutout=False, test=False)

	train_iter = chainer.iterators.SerialIterator(train, batch_size)
	val_iter = chainer.iterators.SerialIterator(val, batch_size)

	
	if resume:
		f = open(resume_pickle,"rb")
		normal = pickle.load(f)

		loss_hist = pickle.load(f)
		train_acc = pickle.load(f)
		model = pickle.load(f)
		advs = pickle.load(f)
		totals = pickle.load(f)
		f.close()

		restart_epoch = int(resume_pickle.split("/")[-1].split(".")[0])
		train_iter.epoch = restart_epoch - 1


	#optimizer1 = chainer.optimizers.MomentumSGD(momentum=0.9)


	optimizer1 = chainer.optimizers.NesterovAG(momentum=0.9)
	optimizer1.setup(model)
	optimizer1.add_hook(chainer.optimizer.WeightDecay(0.0005))
	optimizer1.add_hook(chainer.optimizer.GradientClipping(1.0))


	baseline = EMA(coef=0.05)


	# SNAP SHOT
	start = time.time()
	os.mkdir('./snapshot/'+str(start))
	iter = 0



	best_arc = normal.deter()
	best_val = 999999.


	while train_iter.epoch < max_epoch:

		if train_iter.epoch >= freeze and iter % 2 == 0:
			update_a = True
			update_w = False
		else:
			update_a = False
			update_w = True

		print("train, ", train_iter.epoch, iter, update_w)
		print("val, ", val_iter.epoch, iter, update_a)


		# SNAP SHOT
		if (iter+1) % 70 == 0:
		#try:
			# output log
			log = str(train_iter.epoch)+','+str(loss.data)+','+str((time.time()-start)/60.0)+',mode='+mode+'\r\n'
			f = open('./snapshot/'+str(start)+'/log.txt','a')
			f.write(log)
			f.close()

			print(log)


			# dump architecture
			arc = normal.deter()
			dump_arc(arc, train_iter.epoch, start, "mode")
			dump_arc(best_arc, train_iter.epoch, start, "best", best_val)


			# dump
			f = open('./snapshot/'+str(start)+'/'+str(train_iter.epoch)+'.pickle', 'wb')
			pickle.dump(normal, f)
			pickle.dump(loss_hist, f)
			pickle.dump(train_acc, f)
			pickle.dump(model, f)
			pickle.dump(optimizer1, f)

			#pickle.dump(train_iter, f)
			f.close()


		# anneal down
		epoch = train_iter.epoch
		lrcoef = (math.cos(epoch*math.pi/max_epoch)+1.0)/2.0
		lrcoef2 = (math.cos((epoch-freeze)*math.pi/(max_epoch-freeze))+1.0)/2.0
		optimizer1.lr = 0.1 * lrcoef
		normal.optimizer.alpha = 0.0003 * lrcoef2



		if update_a:
			batch = val_iter.next()
		else:
			batch = train_iter.next()


		x_array, t_array = convert.concat_examples(batch, device=gpu)
		
		## 
		#util2.plot_batch(x_array)

		x = chainer.Variable(x_array)
		t = chainer.Variable(t_array)


		if mode == "N" and train_iter.epoch < freeze:
			a = normal.draw(nasp_random=True)
		else:
			a = normal.draw()

		if mode == "N" and update_w:
			is_nasp_upd_w = True
		else:
			is_nasp_upd_w = False



		y = model(x, a, mode, is_nasp_upd_w)
		model.cleargrads()



		loss = F.softmax_cross_entropy(y,t)
		# trelance
		#loss = -(F.relu(-loss+4)-4)

		loss_hist.append(float(baseline.get()))

		acc = F.accuracy(y,t)
		train_acc.append(float(acc.data))

		loss.backward(retain_grad=True)

		baseline(float(loss.data)) # update baseline

		


		# update
		if update_a:

			if mode == "R":
				normal.update([float(loss.data)-baseline.get()])
			elif mode == "PA":
				normal.update([float(loss.data)])
			else:
				normal.update()

			# stock the best performance arc
			if float(loss.data) < best_val:
				best_val = float(loss.data)
				best_arc = chainer.Variable(a.data.copy())
			

		if update_w:

			print(loss)
			optimizer1.update()




		iter += 1


if __name__ == '__main__':

	main()

