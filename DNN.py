#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
import glob
import os.path
#from scipy import ndimage
import random
import copy
import sys
import time
import math
#import import_cifar

# global
learn_weight = 0.2 #0.058 # 0.2
sigmoid_a = 1.0
is_sigmoid = 1
layer_num = 3

flg1 = 0
img_s_flg = 0
# 慣性項係数
inerta_coef = 0.2 # 0.6

drop_rate = 0.5

# 各種学習中のパラメータを入れる箱
# learn_data
# {
#	id => (int)
# 	layer_num => int()
# 	thres => double()
# 	err  => double()
# 	result => double()
# 	layers => {
#				[
#					ys => vector
#					W => matrix		
#				], ...		
#			}
# }


# input data info
# input_data 
# {
# 	input_arr_size
# 	input_width     # 訓練データbefore?filter後サイズ
# 	input_height
# 		
# }
input_data ={}


TestDataSet = []

def make_data_set(paths, ans):
	global input_data;
	data_set = []
	for f in paths:
		data = {}
		data['img'] = Image.open(f)
		data['ans'] = ans
		data['name'] = f
		data['arr'] = before_filter(data['img'])
		data_set.append(data)

	# 入力画像一般情報	
	input_data['input_arr_size'] = data['arr'].size
	print "input_arr_size = " + str(input_data['input_arr_size'] )
	
	return data_set

# 凡化性能の確認
def test(learn_data, test_data_set):
	err_sum = 0.0
	success = 0
	one_num = 0
	data_num = len(test_data_set)
	is_multi = 0
	if (len(test_data_set[0]['ans']) > 1):
		is_multi = 1

	print "test result"

	for data_set in test_data_set:
		arr = data_set['arr']
		ans = data_set['ans']

		ret = recog(learn_data, arr, ans, is_multi)

		data_set['result'] = ret['result']
		err_sum += ret['err']
		success += ret['success']
		if(success == 0):
			print str(ret['result']) + " : " + str(ans)

		if(is_multi == 0 and ret['ans'][0] == 1.0):
			one_num +=1

	print str(success) + "/" + str(data_num) 
	print "one_num: " + str(one_num)
	print "errs = " + str(err_sum/ data_num) 

# ans : 配列
def recog(learn_data, input_data, ans, is_multi = 0):
	result = forward_all(learn_data, input_data, is_multi)
	err = calc_rss(result, ans)/len(ans)
	success = 0
	# fix me
	# できれば　multi と ２値を統一したい
	if(is_multi == 0):
		pred = 1
		if(result < 0.5):
			pred = 0
		if(pred == ans[0]):
			success = 1
		if(ans[0] == 1.0):
			one_num = 1
		print str(result) + " : " + str(ans[0])

	else:
		max_index = np.argmax(np.array(result))
		if(ans[max_index]):
				success = 1

	return {"pred":pred, "success":success, "err":err, 'result':result, 'ans':ans} 

def calc_optimum_thres(data_list):
	err_min = 1.0
	for data in data_list:
		err = calc_err_by_thres(data_list, data['result'])
		#print errs
		if err_min > err:
			err_min = err
			thres = data['result']

	#thres = 0.4
	return thres

def calc_err_by_thres(data_list, thres):
	err = 0
	for data in data_list:
		if (data['result'] - thres)*data['ans'] < 0: # あるものをないと推測 
			err += 0.5
		elif (data['result'] - thres)*(1.0 - data['ans']) >0: # ないものをあると予測
			err += 1.0
	return err /len(data_list)

# ws を代入　その際にws_delta 初期化
def init_learn_ws(layers, ws, layer_num):
	layers[layer_num]['W'] = ws
	layers[layer_num]['ws_delta'] = np.zeros(ws.shape)

# 外部から呼ばれることを想定
def learn_m(x_data, y_data, class_num = 1, limit_err = 0.02, is_test = 1):
	global input_data
	learn_data = {}
	data_set = []
	test_data_set = []

	input_data['input_arr_size'] = x_data[0].size
	for i in xrange(len(x_data)):
		data = {}
		data['ans'] = [y_data[i]]
		data['arr'] = x_data[i]
		if(is_test == 1 and i > len(x_data) * 0.8):
			test_data_set.append(data)
		else:
			data_set.append(data)
	start = time.time()
	
	learn(learn_data, data_set, limit_err)
	elapsed_time = time.time() - start
	print ("elapsed_time:{0}".format(elapsed_time)) + "[sec]"

	test(learn_data, test_data_set)

	return learn_data

# 外部から呼ばれる想定 バッチ学習　
# x_data: img_arr , y_data : 0 or 1
def batch_learn(learn_data, x_data, y_data):
	global layer_num
	if(learn_data == None): # はじめてよばれたとき、初期化する
		i_learn_data = {}
		i_learn_data['err'] = 0
		i_learn_data['layer_num'] = layer_num
		i_learn_data['thres'] = 0.5
		layers = []
		for i in range(layer_num): 
			layer = {}
			layers.append(layer)

		i_learn_data['input_arr_size'] = x_data.size
		hidden_size = i_learn_data['input_arr_size']/10 #100
		init_learn_ws(layers, np.random.rand(hidden_size, i_learn_data['input_arr_size']+1) - 0.5 , 1) #/(input_data['input_arr_size'] +1)/100.0
		init_learn_ws(layers, np.random.rand(1, hidden_size+1) - 0.5, 2) #/hidden_size/100.0

		i_learn_data['layers'] = layers
		learn_data = i_learn_data

	err = learn_one(learn_data ,{"arr": x_data, "ans": [y_data]})
	#return {"err": err, "learn_data" : learn_data}
	return (err, learn_data)

# 確率的勾配降下法で学習
def learn(learn_data, data_set, limit_err = 0.066): #0.066
	global layer_num, TestDataSet, input_data
	learn_data['err'] = 0
	learn_data['layer_num'] = layer_num
	learn_data['thres'] = 0.5
	layers = []
	for i in range(layer_num): 
		layer = {}
		layers.append(layer)

	hidden_size = input_data['input_arr_size']
	init_learn_ws(layers, np.random.rand(hidden_size, input_data['input_arr_size']+1) - 0.5 , 1) #/(input_data['input_arr_size'] +1)/100.0
	#init_learn_ws(layers, np.random.rand(hidden_size, hidden_size+1) - 0.5, 2) #/hidden_size/100.0
	init_learn_ws(layers, np.random.rand(1, hidden_size+1) - 0.5, 2) #/hidden_size/100.0

	print "init layers"
	print layers


	learn_data['layers'] = layers
	err_sum = limit_err + 1.0

	loop_c = 0

	while err_sum > limit_err:
		crt_err = 0
		max_zero_err = 0.0 # 真の答えが０の時の予測値の最大
		# shuffle
		size = len(data_set)
		index_list = range(size)
		random.shuffle(index_list)
		#index_list = index_list[0:int(size*0.8)]
		success = 0
		one_num = 0
		for index in index_list:
			err = learn_one(learn_data ,data_set[index])
			crt_err += err
			if(data_set[index]['ans'][0] == 0 and max_zero_err < err):
				max_zero_err = err
			
			data_set[index]['pred'] = 0.0
			#print learn_data['result'] 
			if(learn_data['result'] > 0.5):
				data_set[index]['pred'] = 1.0
			if(data_set[index]['ans'][0] == data_set[index]['pred']):
				success += 1 
			if(data_set[index]['ans'][0] == 1.0):
				one_num += 1

		loop_c += 1
		err_sum = crt_err/len(index_list)
		print "err = "
		print err_sum
		print str(success)+"/" + str(len(index_list))
		print one_num

		if(loop_c > 200):
			sys.stderr.write('not solved ')
			break

		print "loop"
		print loop_c

		# 凡化性能と過学習を調べる
		if(loop_c > 200):
			test(learn_data ,TestDataSet)


	#learn_data['thres'] = 0.5 + math.sqrt(max_zero_err) # err 二乗されたものがかえっているためもとに戻す
	print "thres = "
	print learn_data['thres'] 

	# 多変数識別
	# data_set
	#	{
	#		'img' 	=> img
	#		'ans'	=> [0, 0 , 0, 0 .. ,0, 1, 0, .. 0]
	#	}
def learn_multi(learn_data, data_set, limit_err = 0.02):
	global input_data
	learn_data['err'] = 0
	learn_data['layer_num'] = layer_num
	learn_data['thres'] = 0.5
	ans_num = len(data_set[0]['ans']) # データの種類
	layers = []
	for i in range(layer_num): 
		layer = {}
		layers.append(layer)


	init_learn_ws(layers, np.random.rand(500, input_data['input_arr_size']+1)-0.5, 1) #/(input_data['input_arr_size'] +1)/2.0
	init_learn_ws(layers, np.random.rand(ans_num, 501) - 0.5, 2) #/320.0
	learn_data['layers'] = layers
	err_ave = limit_err + 1.0
	loop_c = 0

	while err_ave > limit_err:
		err_sqr = 0
		# shuffle
		size = len(data_set)
		index_list = range(size)
		random.shuffle(index_list)
		success = 0
		for index in index_list:
			err = learn_one(learn_data ,data_set[index])
			err_sqr += err

			max_index = np.argmax(np.array(learn_data['result']))
			if(data_set[index]['ans'][max_index] == 1.0):
				success += 1 

		loop_c += 1
		err_ave = err_sqr/len(index_list)
		print "err = "
		print err_ave
		print str(success)+"/" + str(size)
		if(loop_c > 200):
			sys.stderr.write('not solved ')
			break

		print "loop"
		print loop_c
	return err_ave

# 重み付けws を表示
def print_W(learn_data):
	global layer_num
	for i in range(layer_num - 1):
		print  learn_data['layers'][i+1]['W']

# 一データに対し、学習し、誤差を返す
def learn_one(learn_data, data):
	global flg1
	# print data
	arr = data['arr']
	ans = data['ans']
	is_multi = 0
	if len(ans) > 1:
		is_multi = 1

	result = forward_all(learn_data, arr, is_multi)
	err = calc_rss(result, ans)/len(ans)
	layer_sum = learn_data['layer_num']
	
	# 逆伝播
	for i in range(learn_data['layer_num']-1):
		index = layer_sum - i - 1 # 2,1
		ws_delta = learn_data['layers'][index]['ws_delta']
		if(index == layer_sum - 1):  # 最後尾
			ws = learn_data['layers'][index]['W']
			xs = learn_data['layers'][index-1]['ys']
			if(is_multi):	# ソフトマックスの場合、式が異なる
				delta_vec = copy.copy(result) - np.array(ans)*np.ones(len(ans))
				ret =  _backward(ws, xs, result, delta_vec, ws_delta)
			else:
				ret = backward(ws, xs, learn_data['result'], np.array(ans), ws_delta)
			
			new_ws = ret['new_ws']
			delta_vec = ret['delta_vec']
		else:
			ws = learn_data['layers'][index]['W']
			xs = learn_data['layers'][index - 1]['ys']
			ys = learn_data['layers'][index]['ys']
			child_ws = learn_data['layers'][index + 1]['W']
			child_delta_vec = copy.copy(delta_vec) # とりあえずコピーしておく　必要ないかも
			ret = backward_in_hidden_layer(ws, xs,  ys, child_ws, child_delta_vec, ws_delta)
			learn_data['layers'][index+1]['W'] = copy.copy(new_ws) # 先に前のぶんを更新しておく
			new_ws = ret['new_ws']
			delta_vec = ret['delta_vec']

		learn_data['layers'][index]['ws_delta'] = ret['ws_delta']
	
	learn_data['layers'][1]['W'] = copy.copy(new_ws)

	if(flg1 == 0):
		flg1=1.0
		print 	learn_data['layers']
	return err

def calc_rss(result, ans):
	diff = np.array(result) - np.array(ans)
	return np.dot(diff, diff)

# 一回,回す
def forward_all(learn_data, arr, is_multi = 0):
	layer_sum = learn_data['layer_num']
	learn_data['layers'][0]['ys'] =  copy.copy(arr) # 参照渡し　対策

	for i in range(learn_data['layer_num']-1):
		parent_layer = learn_data['layers'][i]
		layer = learn_data['layers'][i+1]
		if(is_multi == 1 and i+2 == layer_sum ): #複数識別で最後尾は活性化関数をソフトマックスに
			f_arr = forward_without_activation(parent_layer['ys'], layer['W'])
			arr = soft_max(f_arr)
		else:
			arr = forward(parent_layer['ys'], layer['W'])
		layer['ys'] = copy.copy(arr)

	learn_data['result'] = learn_data['layers'][layer_sum -1]['ys']
	
	return learn_data['result']

# 画像前処理
# 256*256 => 64*64 になる
def before_filter(img):
	global img_s_flg, input_data
	im = before_filter_img(img)
	if img_s_flg == 0:
		img.show()
		Image.fromarray(im*256).show()
		img_s_flg = 1
		input_data['input_height'] = Image.fromarray(im).size[1]
		input_data['input_width'] = Image.fromarray(im).size[0]

	arr = convert_arr_from_img(im)
	return arr

def before_filter_img(img):
	global img_s_flg
	c_img = copy.copy(img)
	im = np.array(c_img.convert('L'))/256.0
	#im = max_pooling(im)
	#im = div_img(1.5, im)
	#im = gaussian_img(im)
	#im = ndimage.sobel(im/256.0, axis=1, mode='constant')*256.0
	#im = laplacian(im)
	#im = max_pooling(im)
	#im = div_img(1.0, im)	
	return im

def max_pooling(img):
	h = img.size/2/img[0].size
	w = img[0].size/2
	r_img = np.zeros((h,w))
	for x in range(h):
		for y in range(w):
			r_img[x][y] = max(img[x*2][y*2], img[x*2+1][y*2], img[x*2][y*2+1], img[x*2+1][y*2+1]) 
	return r_img

def median_pooling(img):
	h = img.size/2/img[0].size
	w = img[0].size/2
	r_img = np.zeros((h,w))
	for x in range(h):
		for y in range(w):
			r_img[x][y] = np.median(np.array([img[x*2][y*2], img[x*2+1][y*2], img[x*2][y*2+1], img[x*2+1][y*2+1]])) 
	return r_img

def min_pooling(img):
	h = img.size/2/img[0].size
	w = img[0].size/2
	r_img = np.zeros((h,w))
	for x in range(h):
		for y in range(w):
			r_img[x][y] = min(img[x*2][y*2], img[x*2+1][y*2], img[x*2][y*2+1], img[x*2+1][y*2+1]) 
	return r_img


# 重み配列(2次元)　でarrayから値を計算
def forward(arr, ws):
	return apply_vector(forward_without_activation(arr, ws), activation)	# 活性化関数を通す

def forward_without_activation(arr, ws):
	xs = np.append(arr, 1.0) # 定数項付加
	return np.dot(ws, xs)	


# 行列の各要素に適応
def apply_each(mat, fun):
	# 要素が1つの時、2次元配列にならず、1次元になってしまう対策
	if mat.size == 1:
		return fun(mat)
	for y in xrange(mat[0].size):
		for x in xrange(mat.size/ mat[0].size):
			mat[x,y] = fun(mat[x,y])
	return mat

def get_matrix_width(mat):
	if(mat.size == 1):
		return 1
	if(mat.ndim == 1):
		return mat.size
	return mat[0].size

# vector の各要素に作用
def apply_vector(vec, fun):
	# 要素が1つの時, スカラーになってしまう
	if vec.size == 1:
		return fun(vec)
	for n in xrange(vec.size):
		vec[n] = fun(vec[n])
	return vec


def activation(x):
	global is_sigmoid
	if (is_sigmoid == 1):
		return sigmoid(x)
	return np.maximum(0,x)

# activation の微分を得る ただし引数は sigmoid(x) = y
def dif_activation(y):
	global sigmoid_a, is_sigmoid
	if (is_sigmoid == 1):
		return sigmoid_a * y * (1.0 - y)

	dif = 1.0
	if(y == 0.0):
		dif = 0.0
	return dif

def sigmoid(x, a = 1.0):
	return 1.0/(1.0+np.exp(-x*a))

def soft_max(x_arr):
	parent_e = 0.0
	for x in x_arr:
		parent_e += np.exp(x)
	y_arr = np.empty(x_arr.size)
	index = 0
	for x in x_arr:
		y_arr[index] = np.exp(x)/parent_e
		index+=1.0

	return y_arr

# arr に変換
def convert_arr_from_img(img):
	size = img.size
	arr = img.reshape(1,size)
	arr = arr[0] # 1次元ベクトルに
	return arr

# correct_ys を計算し、ws を更新する 更新されたwsが返る(非破壊的　たぶん) 最後尾のみ
def backward(ws, xs, ys, correct_ys, ws_delta):
	delta_vec = (ys - correct_ys) * apply_vector(ys, dif_activation)
	return _backward(ws, xs, ys, delta_vec, ws_delta)

def backward_in_hidden_layer(ws, xs,  ys, child_ws, child_delta_vec, ws_delta):
	delta_vec = calc_delta_vec_in_hidden_layer(ys, child_ws, child_delta_vec)
	return _backward(ws, xs, ys, delta_vec, ws_delta)

# 引数　ws:重みづけ2次元ベクトル　xs:入力　ys:出力(vector)　correct_xs:正しい出力
# delta: vector 
def _backward(ws, xs, ys, delta_vec, old_ws_delta):
	global learn_weight
	del_mat = np.transpose( get_same_array(delta_vec, xs.size + 1))
	diag_arr = np.diag(np.append(xs,1))
	ws_delta = learn_weight * np.dot(del_mat, diag_arr) + inerta_coef * old_ws_delta 
	new_ws = ws - ws_delta
	return {'new_ws':new_ws, 'delta_vec':delta_vec, 'ws_delta':ws_delta}

# 隠れ層のデルタを計算
def calc_delta_vec_in_hidden_layer(ys, child_ws, child_delta_vec):
	global sigmoid_a
	w_size_in_child_ws = get_matrix_width(child_ws)
	if child_ws.ndim == 1: # 1次元の場合、転置できないため例外対応
		mod_child_ws = child_ws[0:w_size_in_child_ws-1].reshape([w_size_in_child_ws - 1, 1])
	else:
		mod_child_ws = (np.transpose(child_ws))[0:w_size_in_child_ws - 1] # 転置して定数項の部分を削除
	
	return np.dot(mod_child_ws, child_delta_vec) * sigmoid_a* ((np.ones(ys.size) - ys) * ys)

# [[arr],
#  [arr],
# [arr]] の配列を得る
def get_same_array(arr, h):
	if arr.ndim > 1 and arr.size != arr[0].size:
		print "warning!! invalid arr shape "
		sys.stderr.write(str(arr.shape))
	n_arr = np.zeros([h, arr.size])
	n_arr[0:h] = arr
	return n_arr

# 一般の画像から場所を特定する施行
def search(learn_data, data_set):
	global input_data
	w_h = 2
	h_h = 2
	data_num = len(data_set)
	print input_data['input_width']
	print input_data['input_height']
	for data in data_set:
		pos_list = search_one(learn_data, data['img'])
		print pos_list
		flamed_img =  data['img'].convert('RGB')
		for pos_data in pos_list:
			pos1 = {'x': (pos_data['pos']['x']) * w_h, 'y':  (pos_data['pos']['y']) * h_h }
			pos2 = {'x': (pos_data['pos']['x'] + input_data['input_width'])* w_h, 'y': (pos_data['pos']['y'] + input_data['input_height'])*h_h }
			
			add_flame(flamed_img, pos1, pos2)
		flamed_img.show()
	
	return 

# 枠をつけた画像を返す
def add_flame(img, pos1, pos2, color = (255,0,0)):
	print "size"
	print pos1
	print pos2
	print img.size
	y1 = pos1['y']
	y2 = pos2['y'] - 1
	for x0 in xrange(pos2['x'] - pos1['x'] -1):
		x1 = x0 + pos1['x']
		img.putpixel((x1,y1), color)
		img.putpixel((x1, y2), color)

	x1 = pos1['x']
	x2 = pos2['x'] - 1
	for y0 in xrange(pos2['y'] - pos1['y'] -1):
		y1 = y0 + pos1['y']
		img.putpixel((x1,y1), color)
		img.putpixel((x2, y1), color)

	return


# pos_list = [
#	{
#		'pos' => pos
# 		'result' => result
#	}
	
#]
# 
def search_one(learn_data, img):
	pos_list = []
	im_arr = before_filter_img(img)
	im = Image.fromarray(im_arr*256.0)
	width = im.size[0]
	height = im.size[1]
	print width
	print height
	max_result = 0.0
	for y in range(height - input_data['input_height'] + 1):
		for x in range(width - input_data['input_width'] + 1):
			if x < 1:
				continue
		
			sliced_img = im.crop((x, y, x + input_data['input_width'], y+ input_data['input_height']))
			arr = convert_arr_from_img(np.array(sliced_img)/256.0)
			result = forward_all(learn_data, arr)
			
			if max_result < result:
				max_result = result
			pred = 0
			if(result > learn_data['thres']):
				pred = 1
				near_index = get_near_pos({'x':x, 'y': y}, pos_list, input_data['input_width'], input_data['input_height'])
				if(near_index == -1): # 近いのがないときは新規登録

					pos_data = {'pos' : {'x':x, 'y':y}, 'result':result}
					pos_list.append(pos_data)
				else: # 近いのがあるときは比較 
					nearest_pos_data = copy.copy(pos_list[near_index])
					if result > nearest_pos_data['result']: # 更新
						pos_list[near_index] =  {'pos' : {'x':x, 'y':y}, 'result':result}
	
	print "max_result = "
	print max_result
	return pos_list

# pos が与えられた際に一番近いposをpos_listの中から見つける ただし基準以上離れている場合は対象としない
def get_near_pos(pos, pos_list, width, height):
	m_dis = 1000; # 大きい値
	near_index = -1 # 暫定のnearest_pos
	index = 0 
	for pos_data in pos_list:
		o_pos = pos_data['pos']
		_w = abs(pos['x'] - o_pos['x'])
		_h = abs(pos['y'] - o_pos['y'])
		if _w < width  and  _h < height :
			if m_dis > _w + _h:
				near_index = index
				m_dis = _w + _h
		index += 1

	#near_index = -1
	return near_index

# 複数の検出機から最適なものを出す
def multi_recognition():
	global input_arr_size
	multi_num = 3 # multi_num 種類を識別する
	# データセット
	data = import_cifar.import_cifar()
	learn_data_set = []
	test_data_set = []
	for i in xrange(600):
		i=i+10
		if(data['Y_train'][i]< multi_num):
			data_set = {}
			data_set['img'] = import_cifar.get_image_from_cifar(data['X_train'][i])
			data_set['ans'] = [0.0] * multi_num
			data_set['ans'][data['Y_train'][i]] = 1.0
			data_set['arr'] = before_filter(data_set['img'])
			learn_data_set.append(data_set)

		if(data['Y_test'][i]< multi_num):
			data_set = {}
			data_set['img'] = import_cifar.get_image_from_cifar(data['X_test'][i])
			data_set['ans'] = [0.0] * multi_num
			data_set['ans'][data['Y_test'][i]] = 1.0
			data_set['arr'] = before_filter(data_set['img'])
			test_data_set.append(data_set)
	
	# 入力画像一般情報	
	input_data['input_arr_size'] = data_set['arr'].size
	print "input_arr_size = " + str(input_data['input_arr_size'] )
	learn_data = {}
	start = time.time()
	learn_multi(learn_data, learn_data_set)
	elapsed_time = time.time() - start
	print_W(learn_data)
	print ("elapsed_time:{0}".format(elapsed_time)) + "[sec]"
	test(learn_data, test_data_set)


	return 

def one_recognition_cifar():
	global input_arr_size
	# データセット
	data = import_cifar.import_cifar()
	learn_data_set = []
	test_data_set = []
	for i in xrange(600):
		i=i+100
		if(data['Y_train'][i]< 2):
			data_set = {}
			data_set['img'] = import_cifar.get_image_from_cifar(data['X_train'][i])
			data_set['ans'] = [data['Y_train'][i]]
			data_set['arr'] = before_filter(data_set['img'])
			learn_data_set.append(data_set)

		if(data['Y_test'][i]< 2):
			data_set = {}
			data_set['img'] = import_cifar.get_image_from_cifar(data['X_test'][i])
			data_set['ans'] = [data['Y_test'][i]] 
			data_set['arr'] = before_filter(data_set['img'])
			test_data_set.append(data_set)
	
	# 入力画像一般情報	
	input_data['input_arr_size'] = data_set['arr'].size
	print "input_arr_size = " + str(input_data['input_arr_size'] )
	learn_data = {}
	start = time.time()
	learn(learn_data, learn_data_set)
	elapsed_time = time.time() - start
	print_W(learn_data)
	print ("elapsed_time:{0}".format(elapsed_time)) + "[sec]"
	test(learn_data, test_data_set)


def one_recognition():
	global TestDataSet
	# データセット
	#f_paths_k =  glob.glob('./Pictures/teach/koala/*')
	#f_paths_o = glob.glob('./Pictures/teach/other/*')
	f_paths_k = []
	f_paths_o = []
	for i in range(100):
		f_paths_k.append("./CarData/TrainImages/pos-" + str(i) + ".pgm")
		f_paths_o.append("./CarData/TrainImages/neg-" + str(i) + ".pgm")
	data_set = make_data_set(f_paths_k, [1.0])
	data_set +=  make_data_set(f_paths_o, [0.0])

	# 無作為に学習データと検証データを選別
	learn_data_set = []
	test_data_set = []
	data_sum = len(data_set)
	num_list = range(data_sum)
	random.shuffle(num_list)


	# 学習に使う
	learn_sum = 48
	count = 0
	for index in num_list:
		if count < learn_sum:
			learn_data_set.append(data_set[index])
		else:
			test_data_set.append(data_set[index])
		count += 1

	TestDataSet = test_data_set

	# search data set
	search_data_set = []
	for i in range(30):
		path = "./CarData/TestImages/test-" + str(i+20) + ".pgm"
		data = {}
		data['img'] = Image.open(path)
		data['name'] = path
		search_data_set.append(data)

	learn_data = {}

	start = time.time()
	learn(learn_data, learn_data_set)
	elapsed_time = time.time() - start
	print_W(learn_data)
	print ("elapsed_time:{0}".format(elapsed_time)) + "[sec]"
	test(learn_data, test_data_set)

	search(learn_data, search_data_set)
	#test(learn_data_set)
	return



def main():
	#one_recognition()
	multi_recognition()
	#one_recognition_cifar()
	return

#if __name__ == '__main__':
#	main()
