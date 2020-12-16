USE_GPU = True
ZERO = 0.0000000001
ONEHOTTER = ["BI","GD","PA","A","P","R"]


import numpy as np

try:
	import cupy as cp
	import cupy as xp
	from chainer import cuda
except:
	import numpy as xp



import chainer
import chainer.links as L
import chainer.functions as F

import math
import os
import pickle
import time


import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import copy


from chainer.datasets import tuple_dataset



MARKERS = {}
COLORS = {"A":'#1f77b4', "P":'#ff7f0e', "GD":'#2ca02c', "PA":'#d62728', "R":'#9467bd', "S":'#8c564b', "N":'#e377c2', "D":'#7f7f7f', "A2":'#000000'}
LAVELS = {"A":"AdvantageNAS", "P":"ProxylessNAS", "N":"NASP", "GD":"GDAS", "PA":"PARSEC", "S":"SNAS", "R":"REINFORCE", "D":"DARTS", "A2": "AdvantageNAS/w-b"}


