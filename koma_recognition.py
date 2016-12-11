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
import square_space
import pickle

import threading
import sys
import datetime

URL = "http://localhost/flyby/"

HU = 1
KYO = 2
KEI = 3
GIN = 4
KIN = 5
KAK = 6
HIS = 7
OHO = 8

TO = 9
NKYO = 10
NKEI = 11
NGIN = 12
UMA = 13
RYU = 14

# 機械学習クラスとのつなぎこみ用クラス。　駒認識を行う
# 駒８種類の認識　＋　先後識別
class KomaRecognition:

	def __init__(self):

		self.SengoLearnFile = "sengo_recog.txt"
		self.SengoLearnW = None
		self.KomaLearnFile = "koma%d_recog.txt"
		self.KomaLearnW = []
		for i in range(8): # magic number
			self.KomaLearnW.append(None)


		snippet_img_arrs = []
		for i in range(8):
			snippet_img_arrs.append(np.array([]))
		self.snippet_img_arrs = snippet_img_arrs

		self.__row_ave = 0
		self.__col_ave = 0

	def get_probability(self, ban_matrix, pos, koma):
		koma_abs = abs(koma)
		if KomaRecognition.KomaLearnW[koma_abs] == None:
			f = open(KomaRecognition.KomaLearnFile % (koma_abs))
			KomaRecognition.KomaLearnW[koma_abs] = pickle.load(f)
		is_inv = False
		if koma < 0:
			isinv = True

		return self.__get_probability(KomaRecognition.KomaLearnW[koma_abs], ban_matrix, pos,  is_inv)

	def get_sengo_probability(self, ban_matrix, pos):
		if KomaRecognition.SengoLearnW == None:
			f = open(KomaRecognition.SengoLearnFile)
			KomaRecognition.SengoLearnW = pickle.load(f)
		return self.__get_probability(KomaRecognition.SengoLearnW, ban_matrix, pos)

	def __get_probability(self, learn_data, ban_matrix, pos, is_inv = False):
		arr = ban_matrix.get_masu(pos[0], pos[1]).snippet_img_arr
		if is_inv:
			arr = inverse_img(arr)
		ret =  DNN.recog(learn_data, arr, [0])
		return ret['result']

	def set_snippet_img_arr(self, img_arr, target):
		abs_target = target
		if(abs_target < 0):
			abs_target = - target
			self.snippet_img_arrs[abs_target - 1].append(self.inverse_img_arr(img_arr))
		else:
			self.snippet_img_arrs[abs_target - 1].append(img_arr)




	def set_masu_from_one_pic(self, ban_matrix):
		img = ban_matrix.img
		im = np.array(img.convert('L'))
		frame = cv2.Canny(im,80,180) # 10 170
		#frame = cv2.convertScaleAbs(cv2.Laplacian(im, cv2.CV_32F, 8))

		im2 = np.array(frame)
		#Image.fromarray(im2).show()
		line_list = self.get_line_list(im2)
		if line_list == []:
			return 0
		img_list = self.devide_img(im2, line_list)
		#Image.fromarray(add_line(im2, line_list)).show()

		for x in range(9):
			for y in range(9):
				s_img_pos = [line_list[1][x], line_list[0][y]]
				r_img = img_list[x][y]
				masu = square_space.Masu(x, y , r_img, 0, s_img_pos)
				img = self.max_pooling(r_img)
				#if(x==0 and y==0):
					#Image.fromarray(img).show()
				masu.snippet_img_arr = self.convert_arr_from_img(img)/255
				masu.is_koma = self.is_koma(masu.snippet_img_arr)
				ban_matrix.masu_data[x][y] = masu
		return 1


	def add_line(self, img, line_list):
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


	def get_line_list(self, img):
		line_list = []
		width = img.shape[1]
		height = img.shape[0]

		
		row_sum = np.sum(img, axis = 0)
		col_sum = np.sum(img, axis = 1)

		row_mean = row_sum.mean()
		col_mean = col_sum.mean()
		row_limit = row_mean*1.2
		col_limit = col_mean*1.2

		row_list = []
		while (True):
			row_limit += row_mean * 0.1
			limit_size = width/20
			row_list = []
			for i, row in enumerate(row_sum):
				if(row > row_limit):
					if(len(row_list) == 0 or i - row_list[-1] > limit_size): # まだ格納されていないか、距離が離れているときは追加
						row_list.append(i)
					elif(row_sum[row_list[-1]] < row): # 自分のほうが大きい場合は上書き
						row_list[-1] = i
			if len(row_list) <= 10:
				break 
		
		if len(row_list) < 10:
			return []

		col_list = []
		while True:
			col_limit += col_mean*0.1
			limit_size = height/20
			col_list = []
			for i, col in enumerate(col_sum):
				if(col > col_limit):
					if(len(col_list) == 0 or i - col_list[-1] > limit_size): # まだ格納されていないか、距離が離れているときは追加
						col_list.append(i)
					elif(col_sum[col_list[-1]] < col): # 自分のほうが大きい場合は上書き
						col_list[-1] = i
			#print len(col_list)
			if len(col_list) <= 10: 
				break

		if len(col_list) < 10:
			return []

		if(-col_list[0] + 2*col_list[1] - col_list[2] > 4):
			col_list[0] = 2*col_list[1] - col_list[2]
		if(col_list[-1] - 2*col_list[-2] + col_list[-3] > 4):
			col_list[-1] = 2*col_list[-2] - col_list[-3]
				
		line_list.append(row_list)
		line_list.append(col_list)

		return line_list

	# arr に変換
	def convert_arr_from_img(self, img):
		size = img.size
		arr = img.reshape(1,size)
		arr = arr[0] # 1次元ベクトルに
		return arr

	def max_pooling(self, img):
		h = img.size/2/img[0].size
		w = img[0].size/2
		r_img = np.zeros((h,w))
		for x in range(h):
			for y in range(w):
				r_img[x][y] = max(img[x*2][y*2], img[x*2+1][y*2], img[x*2][y*2+1], img[x*2+1][y*2+1]) 
		return  np.uint8(r_img)

	def devide_img(self, img, line_list):
		if(len(line_list[0]) != 10):
			print "err row number in convert_masu_pos" + str(len(line_list[0]))
			return 
		if(len(line_list[1]) != 10):
			print "err col number in convert_masu_pos" + str(len(line_list[1]))
			return 
		
		# 最初と最後のラインは盤の外枠の線にずれる可能性があるので除外
		if(self.__row_ave ==0 or self.__col_ave == 0):
			self.__col_ave = (line_list[1][-2] - line_list[1][1])/7
			self.__row_ave =  (line_list[0][-2] - line_list[0][1])/7

		img_list = []
		for col in range(9):
			row_list = []
			for row in range(9):
				div_img = copy.copy(img[line_list[1][col]:line_list[1][col] + self.__col_ave, line_list[0][row]: line_list[0][row] + self.__row_ave])
				row_list.append(div_img)

			img_list.append(row_list)

		return img_list

	def inverse_img(self, img):
		return  (np.fliplr(img))[::-1]

	def inverse_img_arr(self, arr):
		return arr[::-1]

	def is_koma(self, img_arr):
		trim_size = 4
		img_arr2 = copy.copy(img_arr.reshape([self.__col_ave/2, self.__row_ave/2]))
		img_arr2 = img_arr2[trim_size:self.__col_ave/2 - trim_size, trim_size:self.__row_ave/2 - trim_size]
		#print np.mean(img_arr2)
		if(np.mean(img_arr2) < 0.3):
			return False
		return True




