# -*- coding: utf-8 -*-
import pickle
import sys


labels = ["skip-connect","tanh","sigmoid","relu","zero"]
labels = ["skip-connect","1x1conv","3x3conv","3x3avg-pool","zero"]



### CIFAR
if False:
	genotype = "|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|nor_conv_3x3~0|nor_conv_3x3~1|nor_conv_3x3~2|" # 93.76 Aargmax0 without
	genotype = "|conv_3x3~0|+|conv_3x3~0|conv_3x3~1|+|conv_3x3~0|conv_3x3~1|conv_3x3~2|" # 93.76 Aargmax1 with

else:
#### PTB
#genotype = "|relu~0|+|relu~0|relu~1|+|relu~0|relu~1|relu~2|" # A best test eval 94.74023980029862 90.95861851808046
	genotype = "|relu~0|+|relu~0|zero~1|+|relu~0|relu~1|tanh~2|" # P best test eval 95.00057536680049 90.4163709649887
	genotype = "|relu~0|+|relu~0|relu~1|+|relu~0|tanh~1|relu~2|" # A best val eval 94.65775050997867 91.12076532430376
	genotype = ""


def node_code(num):
	return "node"+str(num)+" [shape = circle,label=\""+str(num)+"\", style=filled, fillcolor = lightblue];\n"

def main(genotype, auto_aggre=True):
	import matplotlib.pyplot as plt

	dot = "digraph g{\n"
	dot += "node [fontsize=30,penwidth=3,fontname=\"Sans Bold\"];\n"
	dot += "edge [arrowsize=3,penwidth=3];\n"


	genotype = genotype.split("+")
	if auto_aggre:
		dot += node_code(len(genotype)+1)

	for i in range(len(genotype)+1):
		dot += node_code(i)
		if auto_aggre and i != 0:
			dot += "node"+str(i)+" -> node"+str(len(genotype)+1)+"\n"


	for i in range(1,len(genotype)+1):
		ops = genotype[i-1].split("|")[1:-1]
		for j in range(len(ops)):
			op,fromn = ops[j].split("~")
			dot += "node"+str(j)+" -> node"+str(j)+str(i)+"[arrowsize=0]\n"
			dot += "node"+str(j)+str(i)+"[shape = rect,label=\""+op+"\"]\n"
			dot += "node"+str(j)+str(i)+" -> node"+str(i)+"[]\n"


	dot += "\n}"


	f = open("dag.dot","w")
	f.write(dot)
	f.close()

	from subprocess import check_call
	check_call(['dot','-Tpng dag.dot','-o','dag.png'])



def arc2query(arc):
	pos = 0
	text = "|"
	for i in range(3):
		for j in range(0,i+1):
			for op_type in range(arc.shape[1]):
				if float(arc[pos,op_type].data) == 1.:
					if op_type == 0:
						text += "skip-connect"
					if op_type == 1:
						text += "tanh"
					if op_type == 2:
						text += "sigmoid"
					if op_type == 3:
						text += "relu"
					if op_type == 4:
						text += "zero"

			text += "~"
			text += str(j)
			text += "|"
			pos += 1
		if i != 2:
			text += "+|"

	return text


offset = 3
def ptb_a_to_genotype(base, mode):


	sys.path.append("ptb")
	import default

	for trial in range(1):
		dump_dir = base+str(trial+offset)
		f = open('./ptb/result/'+dump_dir+'/'+mode+'_eval_model','rb')
		_ = pickle.load(f)
		a = pickle.load(f)
		f.close()

		genotype = arc2query(a)

		print(genotype)

		f = open('./ptb/result/'+dump_dir+'/'+mode+"_eval_log",'r')
		r = f.read()
		f.close()

		r = r.split(" ")
		valp = float(r[3].replace(",",""))
		testp = float(r[6])

		print(valp, testp)
		print("=======")



ptb_a_to_genotype("rb", "A")

#main(genotype, True)
exit()
