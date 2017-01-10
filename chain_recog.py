#!/usr/bin/env python
# -*- coding: utf-8 -*-
from chainer import FunctionSet, Variable, optimizers
import chainer.functions as F
import numpy as np
import random
import matplotlib.pyplot as plt

def _forward(x_data, y_data, model, train= True):
	x, t = Variable(x_data), Variable(y_data)
	h1 = F.dropout(F.relu(model.l1(x)), train = train)
	y  = model.l2(h1)
	return (y,t)

def forward(x_data, y_data, model, train= True):
	(y, t) = _forward(x_data, y_data, model, train)
	return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

def recog(x_data, y_data, model):
	(y, t) = _forward(x_data, y_data, model, train = False)
	ret = []
	for ans_arr in y.data:
		ans_arr = np.array(ans_arr)
		ret.append(ans_arr.argmax())
	print ( F.softmax_cross_entropy(y, t).data, F.accuracy(y, t).data)
	return ret

def learn(x_data, y_data, class_num = 2):

	# 訓練用とtest に分ける
	size = len(x_data)
	N = int(size * 0.8)
	N_test = size - N
	x_train = x_data[0:N]
	y_train = y_data[0:N]
	x_test = x_data[N:size]
	y_test = y_data[N:size]

	train_acc = []
	train_loss = []

	test_acc = []
	test_loss =  []

	l1_W = []
	l2_W = []
	

	input_size = int(x_train[0].shape[0])
	output_size = class_num

	print input_size
	print output_size

	print "data set num = " + str(len(x_train))
	# 多層パーセプトロンの定義
	model = FunctionSet(l1=F.Linear( input_size, input_size*2),
	                    l2=F.Linear(input_size*2, output_size))

	# Setup an optimizer
	optimizer = optimizers.Adam()
	optimizer.setup(model)
	
	# 勾配計算
	# ミニバッチを初期化
	n_epoch = 20
	batchsize = 36
	
	# Learning loop
	for epoch in xrange(1, n_epoch+1):
	    print 'epoch', epoch

	    # training
	    # N個の順番をランダムに並び替える
	    perm = np.random.permutation(N)
	    sum_accuracy = 0
	    sum_loss = 0
	    # 0〜Nまでのデータをバッチサイズごとに使って学習
	    for i in xrange(0, N, batchsize):
	        x_batch = x_train[perm[i:i+batchsize]]
	        y_batch = y_train[perm[i:i+batchsize]]

	        # 勾配を初期化
	        optimizer.zero_grads()
	        # 順伝播させて誤差と精度を算出
	        loss, acc = forward(x_batch, y_batch, model)
	        # 誤差逆伝播で勾配を計算
	        loss.backward()
	        optimizer.update()
	        sum_loss     += loss.data * batchsize
	        sum_accuracy += acc.data * batchsize

	    # 訓練データの誤差と、正解精度を表示
	    print 'train mean loss={}, accuracy={}'.format(sum_loss / N, sum_accuracy / N)

	    train_loss.append(sum_loss / N)
	    train_acc.append(sum_accuracy / N)

	    # evaluation
	    # テストデータで誤差と、正解精度を算出し汎化性能を確認
	    sum_accuracy = 0
	    sum_loss     = 0
	    for i in xrange(0, N_test, batchsize):
	        x_batch = x_test[i:i+batchsize]
	        y_batch = y_test[i:i+batchsize]

	        # 順伝播させて誤差と精度を算出
	        loss, acc = forward(x_batch, y_batch, model, train=False)

	        sum_loss     += loss.data* batchsize
	        sum_accuracy += acc.data * batchsize

	    # テストデータでの誤差と、正解精度を表示
	    print 'test  mean loss={}, accuracy={}'.format(sum_loss / N_test, sum_accuracy / N_test)
	    test_loss.append(sum_loss / N_test)
	    test_acc.append(sum_accuracy / N_test)

	    # 学習したパラメーターを保存
	    l1_W.append(model.l1.W)
	    l2_W.append(model.l2.W)
	  
	return model

	# 精度と誤差をグラフ描画
	plt.figure(figsize=(8,6))
	plt.plot(range(len(train_acc)), train_acc)
	plt.plot(range(len(test_acc)), test_acc)
	plt.legend(["train_acc","test_acc"],loc=4)
	plt.title("Accuracy of digit recognition.")
	plt.plot()