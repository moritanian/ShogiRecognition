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

col_ave = 0
row_ave = 0

# 微分フィルタ
def div_img2(w, img):
	fil = np.array( [[-w , -w, -w],
       			[0, 0, 0],
       			[w, w, w]])
	con = apply_filter(fil, img) #畳み込み
	return con

def laplacian(img):
	fil = np.array( [[1.0 , 1.0, 1.0],
       			[1.0, -8.0, 1.0],
       			[1.0, 1.0, 1.0]])/100.0
	con = apply_filter(fil, img) #畳み込み
	return con

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

	# fixme 丁度１０このラインになるよう閾値自動調整すべき
	row_mean = row_sum.mean()*2
	col_mean = col_sum.mean()*2

	limit_size = width/20
	row_list = []
	for i, row in enumerate(row_sum):
		if(row > row_mean):
			if(len(row_list) == 0 or i - row_list[-1] > limit_size): # まだ格納されていないか、距離が離れているときは追加
				row_list.append(i)
			elif(row_sum[row_list[-1]] < row): # 自分のほうが大きい場合は上書き
				row_list[-1] = i

	limit_size = height/20
	col_list = []
	for i, col in enumerate(col_sum):
		if(col > col_mean):
			if(len(col_list) == 0 or i - col_list[-1] > limit_size): # まだ格納されていないか、距離が離れているときは追加
				col_list.append(i)
			elif(col_sum[col_list[-1]] < col): # 自分のほうが大きい場合は上書き
				col_list[-1] = i

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
	
	
	if(row_ave ==0 or col_ave == 0):
		col_ave = (line_list[1][-1] - line_list[1][0])/9
		row_ave =  (line_list[0][-1] - line_list[0][0])/9

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


def make_data():
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

	x_data = []
	y_data = []
	for i in range(4):
		path = "init_ban" + str(i) + ".jpg"
		im = Image.open(path)
		im = np.array(im.convert('L'))
		frame = cv2.Canny(im,100,200)
		im2 = np.array(frame)
		#im2 = max_pooling(im2)
		line_list = get_line_list(im2)
		img_list = devide_img(im2, line_list)

		learn_masu = [[0,0], [0,1], [0,2], [0,3], [0,4], [0,5], [0,6], [0,7], [0,8], 
							[1,1], [1,7], [2,0], [2,8],
					[8,0], [8,1], [8,2], [8,3], [8,4], [8,5], [8,6], [8,7], [8,8], 
							[7,1], [7,7], [6,0], [6,8], [4,3], [4,4], [4,5], [4,6]]
		
		for masu in learn_masu:
			col = masu[0]
			row = masu[1]
			koma = ban_data[col][row]
			if(koma < 0):
				img = inverse_img(img_list[col][row])
			else:
				img = img_list[col][row]

			#img = max_pooling(img)
			#Image.fromarray(img).show()
			
			x_data.append(convert_arr_from_img(img)/255)

			#y = np.zeros([9])
			#y[int(abs(koma))] = 1
			y =int(abs(koma))
			y_data.append(y)	

	return {"x_data": x_data, "y_data": y_data}


# グレースケール 縮小
im = Image.open("init_ban7.jpg")
im = np.array(im.convert('L'))
#im = laplacian(im)

#im = max_pooling(im)

#img = cv2.imread('init_ban.jpg')
frame = cv2.Canny(im,100,200)
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

learn_data = make_data()
chain_recog.learn(learn_data['x_data'], learn_data['y_data'])

