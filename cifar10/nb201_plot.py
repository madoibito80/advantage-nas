import os

with open(os.path.abspath(__file__).replace("nbplot.py","../setting.py"), "r") as f:
	code = f.read()
	exec(code)




# see https://github.com/D-X-Y/NAS-Bench-201/blob/master/nas_201_api/api.py



import numpy as np
import matplotlib.pyplot as plt


emax = {"D":68, "S":68, "N":76, "A":100}

def opener(path, freeze=50, mode=None):
	lines = []
	f = open(path, "r")
	line = f.readline()
	while line:
		if '|' in line:
			line = line.replace('\n','').split(' ')
			epoch = int(line[0])
			arc = line[4]
			if epoch >= freeze and epoch < emax[mode]:
				lines.append(arc)
		line = f.readline()

	f.close()
	return lines




ways = ["argmax","best"]
modes = ["A","GD","P","R","PA"]
trials = [0,1,2]
label = {"A":"AdvantageNAS", "GD":"GDAS", "P":"ProxylessNAS", "R":"REINFORCE", "PA":"PARSEC"}
label2 = {"best":"PB", "argmax":"AM"}


modes = ["A","D","S","N"]
label = {"D":"DARTS", "S":"SNAS", "N":"NASP", "A":"AdvantageNAS"}
ways = ["argmax"]

def make(ww="without"):

	from nas_201_api import NASBench201API as API
	api = API('/Users/madoibito80/NAS-Bench-201-v1_0-e61699.pth')

	macc = {}
	nparams = {}

	for mode in modes:
		for way in ways:
			if mode == "P" and way == "best":
				continue

			for trial in trials:
				fname = str(trial) + "_" + mode
				arcs = opener("./snapshot/"+ww+"/"+way+"/"+fname+".txt", freeze=50, mode=mode)
				accs = []
				for i in range(len(arcs)):
					if i % 100 == 0:
						print(mode,way,trial,i)
					index = api.query_index_by_arch(arcs[i])
					flg = True
					try:
						info = api.query_meta_info_by_index(index)
					except:
						print("error: ",mode, way, trial, i)
						flg = False

					if flg:
						res = info.get_metrics('cifar10', 'ori-test', None, False)
						# cifar10 : training the model on the CIFAR-10 training + validation set.
						# ,criteria , , False=3average of NAS-Bench)
						acc = res['accuracy']
						accs.append(float(acc))
					else:
						accs.append(accs[-1])


				if trial == 0:
					macc[mode+way] = np.zeros((len(trials),len(accs)))
					nparams[mode+way] = np.zeros((len(trials),1))
				macc[mode+way][trial] = np.array(accs)

				# get final performance for table
				try:
					print(dir(info))
				except:
					print("no dir")
				metric = info.get_comput_costs('cifar10')
				flop, param, latency = metric['flops'], metric['params'], metric['latency']
				nparams[mode+way][trial] = float(param)


	f = open("./"+ww+".pickle","wb")
	pickle.dump(macc, f)
	pickle.dump(nparams, f)
	f.close()

	print(macc)
	print(nparams)
	return macc



#make("without3")
#exit()


def plotter(ww):
	hfont = {'fontname':'Helvetica'} 

	import matplotlib
	matplotlib.rcParams['ps.useafm'] = True
	matplotlib.rcParams['pdf.use14corefonts'] = True

	#matplotlib.rcParams["font.serif"] = "Times"
	#matplotlib.rcParams['text.usetex'] = True
	#matplotlib.rcParams["font.family"] = "Times New Roman"

	#matplotlib.rcParams["font.family"] = "Helvetica"
	#matplotlib.rcParams['ps.useafm'] = True
	#matplotlib.rcParams['pdf.use14corefonts'] = True

	font_size = 9
	plt.rcParams["font.size"] = font_size

	plt.tight_layout()

	f = open("./"+ww+".pickle","rb")
	macc = pickle.load(f)
	f.close()

	if ww == "without":
		plt.figure(figsize=(3.30,1.86))
	if ww == "without3":
		plt.figure(figsize=(3.30,1.6))

	for mode in modes:
		for way in ways:
			if mode == "P" and way == "best":
				continue

			y = np.percentile(macc[mode+way], 50, axis=0)
			#y = np.mean(macc[mode+way], axis=0)


			if mode in MARKERS.keys():
				marker = MARKERS[mode]
			else:
				marker = ""

			if mode == "A" and way == "argmax":
				width = 3
			else:
				if ww == "without3":
					width = 1.5
				if ww == "without":
					width = 1.

			if way == "best":
				style = "--"
			else:
				style = "-"

			if mode in COLORS.keys():
				color = COLORS[mode]
			else:
				color = ""
			print(y)

			if way == "argmax":
				if ww == "without3":
					plt.plot(np.linspace(0,3.56,len(y)),y, label=label[mode], marker=marker, linestyle=style, color=color, lw=width)
				if ww == "without":
					plt.plot(np.linspace(0,50,len(y)),y, label=label[mode], marker=marker, linestyle=style, color=color, lw=width)
			
			# 3.56 GPU-hour/ 50 epoch

	#if ww == "without":
	#	plt.ylim(85.5,94.5)
	#else:
	#	plt.ylim(80,94.5)
	if ww == "without":
		plt.xlim(-0.3,7.5)
		plt.ylim(91.4,93.9)
	else:
		plt.ylim(86.5,94.5)

	plt.grid(b=True)
	if ww == "without":
		plt.xlabel('search epoch')
	if ww == "without3":
		plt.xlabel('GPU-hour')

	if ww == "without":
		plt.ylabel('NAS-Bench-201 \ntest accuracy')
	if ww == "without3":
		plt.ylabel('NAS-Bench-201 \ntest accuracy')


	plt.legend(bbox_to_anchor=(0.5,-0.03),loc='lower center', ncol=2, fontsize=font_size, handlelength=0.85, columnspacing=0.5,handletextpad=0.5,labelspacing=0.29,borderpad=0.3,edgecolor="#ffffff",framealpha=1)

	if ww == "without":
		plt.subplots_adjust(left=0.2, right=0.99, bottom=0.21, top=0.96)
	if ww == "without3":
		plt.subplots_adjust(left=0.17, right=0.98, bottom=0.23, top=0.98)

	plt.savefig("./"+ww+".pdf")#, bbox_inches='tight', pad_inches=0.0)




plotter("without3")
exit()







def plotter2():

	fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8,4.35))
	plt.subplots_adjust(wspace=0.13, hspace=0)

	for ww in ["with","without"]:

		f = open("./"+ww+".pickle","rb")
		macc = pickle.load(f)
		f.close()

		for mode in modes:
			for way in ways:
				if mode == "P" and way == "best":
					continue

				y = np.percentile(macc[mode+way], 50, axis=0)


				if mode in MARKERS.keys():
					marker = MARKERS[mode]
				else:
					marker = ""

				if mode == "A" and way == "argmax":
					width = 5
				else:
					width = 1.5

				if way == "best":
					style = "-."
				else:
					style = "-"

				if mode in COLORS.keys():
					color = COLORS[mode]
				else:
					color = ""

				
				if ww == "with":
					target = ax1
				else:
					target = ax2

				target.plot(np.linspace(0,50,len(y)),y, label=label[mode]+'('+label2[way]+')', marker=marker, linestyle=style, color=color, lw=width)
				


	ax1.set_ylim(84,94.5)
	ax2.set_ylim(84,94.5)


	ax1.grid(b=True)
	ax2.grid(b=True)
	#plt.xlabel('search epoch')
	ax1.set_ylabel('NAS-Bench-201 test accuracy')
	#ax1.legend(loc='lower right')
	ax2.legend(loc='lower right')
	ax1.set_title("with zero operation")
	ax2.set_title("without zero operation")
	#plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.93)


	fig.text(0.5, 0.01, 'search epoch', horizontalalignment='center')#, fontsize=14)
	plt.savefig("./"+ww+".eps", bbox_inches='tight', pad_inches=0.05)





def make_table(ww):

	f = open("./"+ww+".pickle","rb")
	macc = pickle.load(f)
	nparams = pickle.load(f)
	f.close()


	for mode in modes:
		print(mode,ways)
		for way in ways:
			if mode == "P" and way == "best":
				continue

			#print(macc[mode+way][:,-1].shape, nparams[mode+way].shape)
			acc = np.mean(macc[mode+way][:,-1])
			astd = np.std(macc[mode+way][:,-1])

			params = np.mean(nparams[mode+way])
			nstd = np.std(nparams[mode+way])

			acc = round(acc,2)
			astd = round(astd,2)

			params = round(params,2)
			nstd = round(nstd,2)

			acc = str(acc).ljust(5,'0')
			nstd = str(nstd).ljust(4,'0')

			print("$",acc,"\\pm",astd,"$ & $" ,params,"\\pm",nstd, "$")



def choose_best(ww):

	f = open("./"+ww+".pickle","rb")
	macc = pickle.load(f)
	nparams = pickle.load(f)
	f.close()

	best_acc = 0.

	print(nparams)
	print(macc)

	for mode in modes:
		for way in ways:
			print(mode,way)
			if mode == "P" and way == "best":
				continue

			for trial in range(3):
				acc = macc[mode+way][trial,-1]
				if acc > best_acc:
					best_acc = acc
					best = mode+way+str(trial)

	print(best_acc, best)
#choose_best("with")
#make_table("without")



