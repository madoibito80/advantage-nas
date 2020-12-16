
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


with open(os.path.abspath(__file__).replace("entropy.py","../setting.py"), "r") as f:
	code = f.read()
	exec(code)


#from basemodel import *
#import cifar



def without_zero():


	dirs = {"A":["1583932084.0711205","1584081794.2731009","1584103307.7785952"],
			#"A":["1597387571.8611045","1597388431.68836","1597388431.691839"], # sub additional baseline
			"GD":["1583987732.3547301","1583987775.3580754","1584031423.7882843"],
			"R":["1584031683.3147745","1584083128.4603267","1584103806.5296545"],
			"P":["1584046741.5498493","1584046781.6052263","1584083207.8849654"],
			"PA":["1584081645.038157","1584081748.374754","1584104827.585834"]}


	hists = {"A": [], "GD": [], "R": [], "P":[], "PA":[]}


	for t in range(3):
		for k,v in dirs.items():
			for e in range(50):

				if e == 0:
					hists[k].append([])

				f = open('./snapshot/'+str(v[t])+'/'+str(e+50)+'.pickle', 'rb')
				normal = pickle.load(f)#pickle.dump(normal, f)
				param = normal.param
				mu = F.softmax(param)

				ent = -F.mean(F.sum(mu*F.log(mu), axis=1))
				ent = float(ent.data)
				hists[k][t].append(ent)
				print(ent)


	f = open("./entropy_hist.pickle", "wb")
	pickle.dump(hists, f)
	f.close()


def plot():

	f = open("./entropy_hist.pickle", "rb")
	hists = pickle.load(f)
	f.close()


	hfont = {'fontname':'Helvetica'} 

	import matplotlib
	matplotlib.rcParams['ps.useafm'] = True
	matplotlib.rcParams['pdf.use14corefonts'] = True

	
	plt.figure(figsize=(3.30,2))

	for k,v in hists.items():
		x = np.zeros((3,50))
		for t in range(3):
			x[t] = hists[k][t]

		mean = np.mean(x, axis=0).reshape(50)
		std = np.std(x, axis=0).reshape(50)
		upp = mean+std
		low = mean-std

		plt.plot(mean, label=LAVELS[k], color=COLORS[k])
		#plt.fill_between(np.arange(50),upp,low,where=upp>low,alpha=0.35, color=COLORS[k])


	#plt.legend(loc='upper center', bbox_to_anchor=(0.98,1.08), ncol=1,handlelength=0.8,handletextpad=0.3,labelspacing=0.29,edgecolor="#ffffff")

	plt.grid(b=True)
	font_size = 9
	plt.legend(bbox_to_anchor=(0.5,-0.03),loc='lower center', ncol=2, fontsize=font_size, handlelength=0.85, columnspacing=0.5,handletextpad=0.5,labelspacing=0.29,borderpad=0.3,edgecolor="#ffffff",framealpha=1)

	plt.subplots_adjust(left=0.17, right=0.96, bottom=0.22, top=0.96)
	#plt.legend()
	plt.ylim(1.17,1.4)
	plt.xlabel('search epoch')
	plt.ylabel('entropy')
	plt.savefig("./entropy_hist.pdf")



plot()
#without_zero()