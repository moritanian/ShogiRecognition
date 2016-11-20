#!/usr/bin/env python
# -*- coding: utf-8 -*-
from chainer import FunctionSet, Variable, optimizers
import chainer.functions as F
import numpy as np
import random

def forward(x_data, y_data, model):
    x, t = Variable(x_data), Variable(y_data)
    h1 = F.relu(model.l1(x))
    y  = model.l2(h1)
    print y.data
    #return F.softmax_cross_entropy(y, t)
    return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

def learn(x_data, y_data):

	

	input_size = int(x_data[0].shape[0])
	output_size = 9

	print input_size
	print output_size

	print "data set num = " + str(len(x_data))

	print y_data 
	# 多層パーセプトロンの定義
	model = FunctionSet(l1=F.Linear( input_size, input_size*2),
	                    l2=F.Linear(input_size*2, output_size))

	# Setup an optimizer
	optimizer = optimizers.Adam()
	optimizer.setup(model)
	
	# 勾配計算
	# ミニバッチを初期化
	
	for i in range(40):
		# 勾配を初期化
		optimizer.zero_grads()
		
		size = len(x_data)
		index_list = range(size)
		random.shuffle(index_list)
		sum_accuracy = 0.0
		sum_loss = 0.0
		for loop in index_list:
			x = np.empty([1, input_size])
			x[0] = x_data[loop]
			x = x.astype(np.float32)
			y = np.array([y_data[loop]])
			y = y.astype(np.int32)
		
			loss, accuracy = forward(x, y, model)  # 順伝播
			loss.backward()                 # 逆伝播
			optimizer.update()
			sum_loss     += loss.data 
        	sum_accuracy += accuracy.data

		print "loop" + str(i)
		print str(sum_loss/len(x_data) )
		print str(sum_accuracy/ len(x_data))
			