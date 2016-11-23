#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
import glob
import os.path
from scipy import ndimage
import cv2
import copy
import matplotlib.pyplot as plt
from scipy import dot, roll

import chain_recog
import DNN
import pickle

col_ave = 0
row_ave = 0

def apply_filter(fil, img):
	return  ndimage.convolve(img, fil)

def add_line(img, line_list):
	c_img = copy.copy(img)
	old = 0
	for row in line_list[0]: # row
		c_img[:, row] = 255
		#print str(row - old)
		old = row 

	old = 0
	for col in line_list[1]: # col
		c_img[col, :] = 255
		#print str(col - old)
		old = col

	return c_img


def get_line_list(img):
	line_list = []
	width = img.shape[1]
	height = img.shape[0]

	
	row_sum = np.sum(img, axis = 0)
	col_sum = np.sum(img, axis = 1)

	row_mean = row_sum.mean()
	col_mean = col_sum.mean()
	row_limit = row_mean*0.2
	col_limit = col_mean*0.2

	row_list = []
	while len(row_list) != 10:
		print len(row_list)
		row_limit += row_mean * 0.1
		limit_size = width/20
		row_list = []
		for i, row in enumerate(row_sum):
			if(row > row_limit):
				if(len(row_list) == 0 or i - row_list[-1] > limit_size): # まだ格納されていないか、距離が離れているときは追加
					row_list.append(i)
				elif(row_sum[row_list[-1]] < row): # 自分のほうが大きい場合は上書き
					row_list[-1] = i
	

	col_list = []
	while len(col_list) != 10:
		print len(col_list)
		col_limit += col_mean*0.03
		limit_size = height/20
		col_list = []
		for i, col in enumerate(col_sum):
			if(col > col_limit):
				if(len(col_list) == 0 or i - col_list[-1] > limit_size): # まだ格納されていないか、距離が離れているときは追加
					col_list.append(i)
				elif(col_sum[col_list[-1]] < col): # 自分のほうが大きい場合は上書き
					col_list[-1] = i

	if(-col_list[0] + 2*col_list[1] - col_list[2] > 4):
		col_list[0] = 2*col_list[1] - col_list[2]
	if(col_list[-1] - 2*col_list[-2] + col_list[-3] > 4):
		col_list[-1] = 2*col_list[-2] - col_list[-3]
			
	line_list.append(row_list)
	line_list.append(col_list)

	return line_list

# arr に変換
def convert_arr_from_img(img):
	size = img.size
	arr = img.reshape(1,size)
	arr = arr[0] # 1次元ベクトルに
	return arr

def max_pooling(img):
	h = img.size/2/img[0].size
	w = img[0].size/2
	r_img = np.zeros((h,w))
	for x in range(h):
		for y in range(w):
			r_img[x][y] = max(img[x*2][y*2], img[x*2+1][y*2], img[x*2][y*2+1], img[x*2+1][y*2+1]) 
	return  np.uint8(r_img)

def devide_img(img, line_list):
	global row_ave, col_ave
	if(len(line_list[0]) != 10):
		print "err row number in convert_masu_pos" + str(len(line_list[0]))
		return 
	if(len(line_list[1]) != 10):
		print "err col number in convert_masu_pos" + str(len(line_list[1]))
		return 
	
	# 最初と最後のラインは盤の外枠の線にずれる可能性があるので除外
	if(row_ave ==0 or col_ave == 0):
		col_ave = (line_list[1][-2] - line_list[1][1])/7
		row_ave =  (line_list[0][-2] - line_list[0][1])/7

	img_list = []
	for col in range(9):
		row_list = []
		for row in range(9):
			div_img = copy.copy(img[line_list[1][col]:line_list[1][col] + col_ave, line_list[0][row]: line_list[0][row] + row_ave])
			row_list.append(div_img)

		img_list.append(row_list)

	return img_list

def inverse_img(img):
	return  (np.fliplr(img))[::-1]

def ban_init():
	HU = 1
	KYO = 2
	KEI = 3
	GIN = 4
	KIN = 5
	KAK = 6
	HIS = 7
	OHO = 8

	ban_data = np.zeros([9,9])
	ban_data[0][0] = -KYO
	ban_data[0][1] = -KEI
	ban_data[0][2] = -GIN
	ban_data[0][3] = -KIN
	ban_data[0][4] = -OHO
	ban_data[0][5] = -KIN
	ban_data[0][6] = -GIN
	ban_data[0][7] = -KEI
	ban_data[0][8] = -KYO

	ban_data[8][0] = KYO
	ban_data[8][1] = KEI
	ban_data[8][2] = GIN
	ban_data[8][3] = KIN
	ban_data[8][4] = OHO
	ban_data[8][5] = KIN
	ban_data[8][6] = GIN
	ban_data[8][7] = KEI
	ban_data[8][8] = KYO
	
	ban_data[1][1] = - HIS
	ban_data[1][7] = -KAK
	ban_data[7][1] = KAK
	ban_data[7][7] = HIS

	ban_data[2,:] = -HU
	ban_data[6, :] = HU

	return ban_data


def print_ban(ban_data):
	HU = 1
	KYO = 2
	KEI = 3
	GIN = 4
	KIN = 5
	KAK = 6
	HIS = 7
	OHO = 8

	strl = ""
	print " 9  8  7  6  5  4  3  2  1"
	print "---------------------------"
	for y in xrange(9):
		strl = ""
		for x in xrange(9):
			if(ban_data[y][x] < 0):
				strl += str(int(ban_data[y][x]))
			elif (ban_data[y][x] > 0):
				strl += " " + str(int(ban_data[y][x]))
			else:
				strl += "  "
			strl += " "
		strl += "|" + str(y+1)
		print strl

def ban_recog(learn_data):
	ban_data = ban_init()
	im = Image.open("init_ban19.jpg")
 	dat = make_data_from_one_pic(im, ban_data)
 	for x in xrange(9):
 		for y in xrange(9):
 			index = x*9 + y
 			ret = DNN.recog(learn_data, dat['x_data'][index], [dat['y_data'][index]])
 			ban_data[x][y] = ret['pred']
 			#ban_data[x][y] = is_koma(dat['x_data'][index])

 	print_ban(ban_data)
	return

def make_data_from_one_pic(img, ban_data, learn_masu = [], learn_koma = 0):
	if(learn_masu == []):
		for i in range(9):
			for j in range(9):
				learn_masu.append([i,j])

	im = np.array(img.convert('L'))
	#frame = cv2.Canny(im,10,170)
	frame = cv2.convertScaleAbs(cv2.Laplacian(im, cv2.CV_32F, 8))

	im2 = np.array(frame)
	Image.fromarray(im2).show()
	line_list = get_line_list(im2)
	img_list = devide_img(im2, line_list)
	
	x_data = []
	y_data = []
	for masu in learn_masu:
		col = masu[0]
		row = masu[1]
		koma = ban_data[col][row]
		if(koma < 0):
			img = inverse_img(img_list[col][row])
		else:
			img = img_list[col][row]

		img = max_pooling(img)
		x_data.append(convert_arr_from_img(img)/255)
		y =int(abs(koma))
		if(learn_koma > 0):
			if(y==learn_koma):
				y=1
			else:
				y=0
		elif(learn_koma == -1):
			if(y!=0):
				y=1
			else:
				y = 0
		elif(learn_koma == -2):
			if(koma < 0):
				y = 0
			else:
				y = 1
		y_data.append(y)	
	return {'x_data': x_data, 'y_data': y_data}

def is_koma(img_arr):
	global col_ave, row_ave
	trim_size = 4
	img_arr2 = copy.copy(img_arr.reshape([col_ave/2, row_ave/2]))
	img_arr2 = img_arr2[trim_size:col_ave/2 - trim_size, trim_size:row_ave/2 - trim_size]
	print np.mean(img_arr2)
	if(np.mean(img_arr2) < 0.3):
		return 0
	return 1

def make_data():
	HU = 1
	KYO = 2
	KEI = 3
	GIN = 4
	KIN = 5
	KAK = 6
	HIS = 7
	OHO = 8
	ban_data = ban_init()

	print_ban(ban_data)

	x_data = []
	y_data = []

	learn_koma = -2;
	file_num = 1
	if(learn_koma == 0):
		file_num = 4
		learn_masu = [[0,0], [0,1], [0,2], [0,3], [0,4], [0,5], [0,6], [0,7], [0,8], 
							[1,1], [1,7], [2,0], [2,8],
					[8,0], [8,1], [8,2], [8,3], [8,4], [8,5], [8,6], [8,7], [8,8], 
							[7,1], [7,7], [6,0], [6,8], [4,3], [4,4], [4,5], [4,6]]
	elif(learn_koma == HU):
		file_num = 20
		learn_masu = [[0,0],[1,0],[2,0],[6,0], [7,0], [8,0], [0,2],[1,2],[2,2],[6,2], [7,2], [8,2]]

	elif(learn_koma == KEI):
		file_num = 20
		learn_masu = [[0,1],[1,1],[2,1],[6,1], [7,1], [8,1], [0,7],[1,7],[2,7],[6,7], [7,7], [8,7]]

	elif(learn_koma == -1):	# あるかないか
		file_num = 19
		learn_masu = [[1,0],[1,1],[1,2],[1,3],[1,4],[1,5],[1,6],[1,7],[1,8],
						[2,0],[2,1],[2,2],[2,3],[2,4],[2,5],[2,6],[2,7],[2,8],
						[3,0],[3,1],[3,2],[3,3],[3,4],[3,5],[3,6],[3,7],[3,8]
						 ]
	elif(learn_koma == -2): # 先後
		file_num = 19
		learn_masu = []
		for i in range(9):
			learn_masu.append([0,i])
			learn_masu.append([2,i])
			learn_masu.append([6,i])
			learn_masu.append([7,i])
	

	cnt = 0
	for i in range(file_num):
		path = "init_ban" + str(i) + ".jpg"
		im = Image.open(path)
		ret = make_data_from_one_pic(im, ban_data, learn_masu , learn_koma )
		x_data += ret['x_data']
		y_data += ret['y_data']

	return {'x_data': x_data, 'y_data': y_data}



def test():
	# グレースケール 縮小
	im = Image.open("init_ban1.jpg")
	im = np.array(im.convert('L'))
	#im = laplacian(im)

	#im = max_pooling(im)

	#img = cv2.imread('init_ban.jpg')
	frame = cv2.Canny(im,10,180)
	#cv2.imshow('image', frame)


	im2 = np.array(frame)
	#im2 = max_pooling(im2)

	line_list = get_line_list(im2)
	line_img = add_line(im2, line_list)


	Image.fromarray(line_img).show()
	img_list = devide_img(im2, line_list)
	Image.fromarray(inverse_img( img_list[0][0]) ).show()



	#hist, bins = np.histogram( line_img, bins=256 )

	# 0〜256までplot
	#plt.plot( hist )
	#plt.xlim(0, 256)
	#plt.show()

#test()
learn_data = make_data()
#chain_recog.learn(learn_data['x_data'], learn_data['y_data'])

learn_data =  DNN.learn_m(learn_data['x_data'], learn_data['y_data'])
f = open('learn_data.txt', 'w')
pickle.dump(learn_data, f)
f.close()
#f = open('learn_data.txt')
#learn_data = pickle.load(f)

ban_recog(learn_data)