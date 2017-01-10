#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
from scipy import ndimage
import cv2
import copy
import random

import DNN
import square_space
import pickle

import threading
import sys
import datetime
import math

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

# 画像処理及び機械学習クラスとのつなぎこみ用クラス。　駒認識を行う
# 駒８種類の認識　＋　先後識別
class KomaRecognition:

	def __init__(self, is_learning = False, log_obj = None):
		self.flg = 0

		self.log_obj = log_obj
		self.is_learning = is_learning # 進行形で学習を行うか

		self.SengoLearnFile = "sengo_recog.txt"
		self.SengoLearnW = None
		self.KomaLearnFile = "learn_data/koma%d_recog.txt"
		self.KomaLearnW = []
		for i in range(8): # magic number
			self.KomaLearnW.append(None)


		snippet_img_arrs = []
		for i in range(8):
			snippet_img_arrs.append([])
		self.snippet_img_arrs = snippet_img_arrs

		self.__row_ave = 0
		self.__col_ave = 0

		self.__piece_width_ave = 0

		self.__save_image_num = 0 # 画像保存用の関数でしようするファイル名のindex
		#row_ave: 42
		#col_ave: 46

		# board position　処理に数え上げ方式（hough変換なし）を使う
		self.__board_position_old = False



	def get_probability(self, ban_matrix, pos, koma):
		koma_abs = abs(koma)
		if self.KomaLearnW[koma_abs  -1] == None:
			self.KomaLearnW[koma_abs -1] = self.load_learn_data(koma_abs)
		is_inv = False
		if koma < 0:
			isinv = True

		return self.__get_probability(self.KomaLearnW[koma_abs], ban_matrix, pos,  is_inv)

	def get_sengo_probability(self, ban_matrix, pos):
		if self.SengoLearnW == None:
			f = open(self.SengoLearnFile)
			self.SengoLearnW = pickle.load(f)
		return self.__get_probability(self.SengoLearnW, ban_matrix, pos)

	def dump_learn_data(self, learn_data, koma):
		koma_abs = koma
		if(koma_abs <0):
			koma_abs = -koma_abs
		f = open(self.KomaLearnFile % (koma_abs- 1), 'w')
		pickle.dump(learn_data, f)
		f.close()

	def load_learn_data(self, koma):
		koma_abs = koma
		if(koma_abs < 0):
			koma_abs = -koma
		if(koma_abs > 8):
			return None
		f = open(self.KomaLearnFile % (koma_abs- 1))
		learn_data = pickle.load(f)
		f.close()
		return learn_data

	def __get_probability(self, learn_data, ban_matrix, pos, is_inv = False):
		arr = ban_matrix.get_masu(pos[0], pos[1]).snippet_img_arr
		if is_inv:
			arr = self.inverse_img_arr(arr)
		ret =  DNN.recog(learn_data, arr, [0])
		return ret['result']

	# 盤面進行時に呼び出される想定 
	# 盤面データから画像を蓄え、非同期に学習していく
	# ban_matrix : 画像をとりこんで処理したやつ
	# crt_ban_matrix : 手番進行しているやつ
	def set_from_banmatrix(self, ban_matrix, crt_ban_matrix):
		# 指定枚数だけ盤面から駒を取得し、学習用データ配列に追加
		take_nums =  [4,4,4,4,4,2,2,2] # 歩、香、桂、銀、金、角、飛、王or 玉
		
		taken_flg = False
		x_list = range(9)
		y_list = range(9)
		random.shuffle(x_list)
		random.shuffle(y_list)
		for x in x_list:
			if taken_flg:
				break
			for y in y_list:
				if taken_flg:
					break
				target = crt_ban_matrix.get_koma(x,y)
				abs_target = abs(target)
				if(abs_target != 0 and abs_target < 9 and take_nums[abs_target - 1] > 0):
					img_arr = self.get_koma_img(ban_matrix, x, y)
					self.__set_koma_img_arr(img_arr, target)
					take_nums[abs_target - 1] -= 1
					
					if take_nums.count(0) == len(take_nums):
						taken_flg = True

		# 別スレッドで学習
		self.learn_thread = threading.Timer(0, self.learn_seq) 
		self.learn_thread.start()

	# セットされた学習用データからランダムにきめられた数だけとりだして学習(１手ごと)
	def learn_seq(self):
		self.log_obj.log("learn seq")

		# ８種類学習
		loop_in_each = 10 # 1種類あたり何枚学習するか
		false_ratio = 0.7 # y_data == 0 になる割合
		for i in range(8):
			for count in range(loop_in_each):
				err_sum = 0
				ans = 1
				if(random.random() < false_ratio):
					ans = 0
				if(ans == 1): # 
					size = len(self.snippet_img_arrs[i])
					index = int(random.random()*size)
					x_data = self.snippet_img_arrs[i][index]

				else: # ほかの駒
					# まずはどの駒かランダムに決める
					target = int(random.random()*7) # 0~7
					if(i <= target ):
						target+= 1 
					size = len(self.snippet_img_arrs[target])

					index = int(random.random()*size)
					x_data = self.snippet_img_arrs[target][index]
					if(self.flg == 1):
						self.flg = 2
						print target+1
						self.show_from_arr(x_data)
				
				(err, self.KomaLearnW[i]) = DNN.batch_learn(self.KomaLearnW[i], x_data, ans)
				err_sum += err
			self.log_obj.log(str(err_sum))
		self.log_obj.log("end learn seq")
		self.test_learn()

	def test_learn(self):
		loop_num = 10
		target = 1
		index = 0
		test_data_set = []
		success = 0
		for i in range(loop_num):
			if(len(self.snippet_img_arrs[target-1]) <= index):
				target+=1
				index = 0
			if(target == 9):
				loop_num = i
				break
			ret = DNN.recog(self.KomaLearnW[target - 1], self.snippet_img_arrs[target -1][index], [1])
			if ret["success"] == 1:
				success += 1
			index += 1

		self.log_obj.log("test_learn")
		self.log_obj.log(str(success) + "/" +  str(loop_num))



	def __set_koma_img_arr(self, img_arr, target):
		abs_target = target
		if(abs_target < 0):
			abs_target = - target
			self.snippet_img_arrs[abs_target - 1].append(self.inverse_img_arr(img_arr))
		else:
			self.snippet_img_arrs[abs_target - 1].append( img_arr)

	# 注意！！！　masuデータ上書きされる
	# 画像の1/4サイズでこまの有り無し判定
	def set_masu_from_one_pic(self, ban_matrix):
		img = copy.copy(ban_matrix.img)
		im = np.array(img.convert('L'))
		frame = cv2.Canny(im,80,180) # 10 170
		#frame = cv2.convertScaleAbs(cv2.Laplacian(im, cv2.CV_32F, 8))

		im2 = np.array(frame)

		ban_matrix.edge_img = copy.copy(im2)
		
		if(self.__board_position_old):
			line_list = self.get_line_list(im2)
			if line_list == []:
				return 0

			img_list = self.devide_img(im2, line_list)
			ban_matrix.edge_abspos = [line_list[1][1] - self.__col_ave, line_list[0][1] - self.__row_ave]

			for x in range(9):
				for y in range(9):
					s_img_pos = [line_list[1][x], line_list[0][y]]
					r_img = img_list[x][y]
					masu = square_space.Masu(x, y , r_img, 0, s_img_pos)
					img = r_img
					self.flg += 1
					xt = 7
					yt = 9
					if(self.flg == (xt -1) * 9 + yt):			
						Image.fromarray(img).show()

					masu.is_koma = self.is_koma(self.max_pooling(r_img)/255)
					ban_matrix.masu_data[x][y] = masu
		else:
			board_position_list = self.get_board_position(im2)
			if(board_position_list == None):
				return 0

			img_list = self.devide_img_from_board_list(im2, board_position_list)
			# 2017/01/07　2016/12/25以降ではピクセル単位の座標指定は右方向をx, 下方向をyにしていたが、
			# edge_abspos に関しては前にあわせて逆にする(扇検出にのみ使っている模様)
			ban_matrix.edge_abspos = [board_position_list[0][0][0][1], board_position_list[0][0][0][0]]

			for x in range(9):
				for y in range(9):
					r_img = img_list[x][y]
					#if(x == 2 and y == 2):			
					#	Image.fromarray(r_img).show()

					s_img_pos = [board_position_list[x][y][0][0], board_position_list[x][y][0][1]]
					masu = square_space.Masu(x, y , r_img, 0, s_img_pos)
					masu.is_koma = self.is_koma(self.max_pooling(r_img)/255)
					ban_matrix.masu_data[x][y] = masu

			self.board_position_list = board_position_list

		return 1

	def add_line(self, img, line_list):
		c_img = copy.copy(img)
		old = 0
		for row in line_list[0]: # row
			c_img[:, row] = 255
			old = row 

		old = 0
		for col in line_list[1]: # col
			c_img[col, :] = 255
			old = col

		return c_img

	def draw_frame(self, img, start, goal, wid = 1):
		return cv2.rectangle(img,(start[0], start[1]),(goal[0],goal[1]),(0,255,0),wid)


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
			print " row over " + str(len(row_list))
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
			if len(col_list) <= 10: 

				break

		if len(col_list) < 10:
			print " row over " + str(len(col_list))
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
			self.__col_ave = (line_list[1][-2] - line_list[1][1])/7.0
			self.__row_ave =  (line_list[0][-2] - line_list[0][1])/7.0

			if(self.log_obj):
				self.log_obj.log("row_ave: " + str(self.__row_ave), 3)
				self.log_obj.log("col_ave: " + str(self.__col_ave), 3)


		img_list = []
		for col in range(9):
			row_list = []
			for row in range(9):
				div_img = copy.copy(img[line_list[1][col]:line_list[1][col] + int(self.__col_ave), line_list[0][row]: line_list[0][row] + int(self.__row_ave)])
				row_list.append(div_img)

			img_list.append(row_list)

		return img_list

	# 交点リストから分割
	# board_list [x][y] = [[x1, y1], [x2, y2]]
	def devide_img_from_board_list(self, img, board_list):
		img_list = []
		for x in range(9):
			row_list = []
			for y in range(9):
				pos = board_list[x][y]
				#div_img = copy.copy(img[pos[0][0]:pos[1][0], pos[0][1]:pos[1][1]])
				div_img = copy.copy(img[pos[0][1]:pos[1][1], pos[0][0]:pos[1][0]])
				row_list.append(div_img)
			img_list.append(row_list)
		return img_list			

	def inverse_img(self, img):
		return  (np.fliplr(img))[::-1]

	def inverse_img_arr(self, arr):
		return arr[::-1]

	def is_koma(self, img_arr):
		trim_size = 4
		#img_arr2 = copy.copy(img_arr.reshape([self.__col_ave/2, self.__row_ave/2]))
		img_arr2 = img_arr
		img_arr2 = img_arr2[trim_size:int(self.__col_ave/2) - trim_size, trim_size:int(self.__row_ave/2) - trim_size]

		if(np.mean(img_arr2) < 0.3):
			return False
		return True

	def show_from_arr(self, arr, is_koma = False):
		img = self.get_img_from_arr(arr, is_koma)
		print "show from arr"
		print img.shape
		Image.fromarray(img).show()
		
	def get_img_from_arr(self, arr, is_koma = False):
		if(is_koma):
			#return arr.reshape([int(self.__col_ave) - self.__cut_off, self.__cut_width]) * 255
			return arr.reshape([int(self.__cut_height) ,int(self.__cut_width)]) * 255
		else:
			return arr.reshape([int(self.__col_ave), int(self.__row_ave)]) * 255

	# device で指し示された場所取得 advanced_img:盤面を動かした際に認識した画像 
	# newest_ban_mastrix:とりこまれた最新のmaxtrix(扇子「が含まれる想定)
	def get_pointed_pos(self, advanced_img, newest_ban_matrix):
		abspos = self.get_tip_abspos(advanced_img, newest_ban_matrix.img, newest_ban_matrix.edge_abspos)
		
		pos = None
		print abspos
		print newest_ban_matrix.edge_abspos
		if(abspos):
			x = int((abspos[0] - newest_ban_matrix.edge_abspos[0])/ self.__col_ave)
			y = int((abspos[1] - newest_ban_matrix.edge_abspos[1])/self.__row_ave)
			if(0<=x and x < 9 and 0<= y and y< 9):
				pos = [x,y]
		return pos

	# 盤面切り出し
	def cut_img_from_matrix(self, img, edge_abspos):
		ban_region = (edge_abspos[1], edge_abspos[0],  edge_abspos[1] + 9*self.__row_ave, edge_abspos[0] + 9*self.__col_ave)
		im = img.crop(ban_region)
		return im

	# canny による駒移動処理の前処理 差分での判定(cannyより重い気がするので、後処理のほうがいいかも)　手などの障害物が含まれる場合、ここではじく(駒移動判定しない)
	def judge_img_diff(self, advanced_img, newest_ban_matrix):
		ans = {"result" : True, "diff_masu": []}
		im1 = self.cut_img_from_matrix(advanced_img, newest_ban_matrix.edge_abspos)
		im2 = self.cut_img_from_matrix(newest_ban_matrix.img, newest_ban_matrix.edge_abspos)
		diff_img ,  im_edge= self.bg_diff(im1, im2) # 差分抽出
		nLabels, labelImage = cv2.connectedComponents(im_edge) # ラベリング

		for i in range(nLabels  -1):
			a = np.where(labelImage == i+1)
			xs = np.array(a[0])
			ys = np.array(a[1])
			x_len = xs.max() - xs.min()
			y_len = ys.max() - ys.min()
			if(x_len > 150 or y_len > 150):
				self.log_obj.warning("judge_ban len err", 3)
				ans["result"] = False
				print "nlabels = " + str(nLabels)
				return ans
			x = int(xs.mean()/ self.__col_ave)
			y = int(ys.mean()/ self.__row_ave)
			ans["diff_masu"].append([x, y])

		return ans


	# 扇子の先端位置取得
	# 差分を利用
	def get_tip_abspos(self, back_img, img, edge_abspos):
			# 盤面範囲に分割
		if(self.__row_ave == 0):
			self.__row_ave = 42
			self.__col_ave = 46
		ban_region = (edge_abspos[1], edge_abspos[0],  edge_abspos[1] + 9*self.__row_ave, edge_abspos[0] + 9*self.__col_ave)
		
		im = back_img.crop(ban_region)
		im2 = img.crop(ban_region)

		path = "temp/back_img2.jpg"	
		self.save_image(np.array(im2), path)

		im3,  im_edge = self.bg_diff(im, im2)

		find = np.where(im3 == 0)
	
		size = None
		if find:
			size = find[0].size
		print ("size rate=" + str(size * 1.0 /(im3.shape[0] * im3.shape[1])))
		print find
		find[1][np.isnan(find[1])] = 0
		find[0][np.isnan(find[0])] = 0

		path = "temp/bgdiff.jpg"	
		self.save_image(np.array(im_edge), path)

		if(not(math.isnan(size)) and size):
			y = int(find[1][0:size/100].mean())
			x = int(find[0][0:size/100].mean())
			
			cv2.circle(im3,(y, x),2,(0,0,255),3)
			
		else :
			
			return None
		
		return [x + edge_abspos[0] , y + edge_abspos[1]]

	def get_tip_abspos_old(self, back_img, img, edge_abspos):
			# 盤面範囲に分割
		if(self.__row_ave == 0):
			self.__row_ave = 42
			self.__col_ave = 46
		ban_region = (edge_abspos[1], edge_abspos[0],  edge_abspos[1] + 9*self.__row_ave, edge_abspos[0] + 9*self.__col_ave)
		im = back_img.crop(ban_region)
		im2 = img.crop(ban_region)

		self.bg_diff(im, im2)
	
		im = np.array(im.convert('L'))
		im2 = np.array(im2.convert('L'))

		im = cv2.Canny(im,80,180) # 10 170
		im2 = cv2.Canny(im2,80,180) # 10 170

		
		im3 = im - im2
		path = "sample.jpg"
		#cv2.imwrite(path, im3)
		
		kernel = np.ones((3,3),np.uint8)
		#im3 = cv2.erode(im3,kernel,iterations = 1)
		#im3 = cv2.dilate(im3,kernel,iterations = 1) * 20
		im3 = cv2.morphologyEx(im3, cv2.MORPH_OPEN, kernel)
		#im3 = cv2.morphologyEx(im3, cv2.MORPH_CLOSE, kernel)
		im3 = im3 *10

		path = "sample.jpg"
		#cv2.imwrite(path, im3)

		#path = "sample.jpg"
		#cv2.imwrite(path, im3)

		im3 = cv2.adaptiveThreshold(im3,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
	
		#im3 = cv2.morphologyEx(im3, cv2.MORPH_CLOSE, kernel)
		#im3 = cv2.morphologyEx(im3, cv2.MORPH_CLOSE, kernel)

		find = np.where(im3 == 0)
		size = None
		if find:
			size = find[0].size
		print ("size rate=" + str(size * 1.0 /(im3.shape[0] * im3.shape[1])))
		print find
		find[1][np.isnan(find[1])] = 0
		find[0][np.isnan(find[0])] = 0
		if(not(math.isnan(size)) and size):
			y = int(find[1][0:size/100].mean())
			x = int(find[0][0:size/100].mean())
			
			path = "sample.jpg"
			cv2.circle(im3,(y, x),2,(0,0,255),3)
			#self.labeling(im3)
		
			#cv2.circle(im3,(143, 30),2,(0,0,255),3)
			#cv2.imwrite(path, im3)
		else :
			
			return None
		
		return [x + edge_abspos[0] , y + edge_abspos[1]]

	def labeling(self, img):
		nLabels, labelImage = cv2.connectedComponents(img)
		print "nlabels = " + str(nLabels)
		colors = []
		(height, width) = img.shape
		for i in range(1, nLabels + 1):
			colors.append(random.randint(0, 255))
		for y in range(0, height):
			for x in range(0, width):
				if labelImage[y, x] > 0:
					img[y, x] = colors[labelImage[y, x]]
				else:
					img[y, x] = 255
		
		for i in range(nLabels  -1):
			a = np.where(labelImage == i+1)
			xs = np.array(a[0])
			ys = np.array(a[1])
			x_len = xs.max() - xs.min()
			y_len = ys.max() - ys.min()
			if(x_len > 150 or y_len > 150):
				continue
			x = int(xs.mean())

			y = int(ys.mean())
			cv2.circle(img,(y, x),2,(0,0,255),3)

		path = "sample.jpg"
		cv2.imwrite(path, img)


	def bg_diff(self, im, im2):
		im = np.array(im.convert('L'))
		im2 = np.array(im2.convert('L'))
		diff = cv2.absdiff(im,im2)
		# 差分が閾値より小さければTrue
		th = 20
		blur = 7
		mask = diff < th
		# 配列（画像）の高さ・幅
		hight = im.shape[0]
		width = im2.shape[1]
		# 背景画像と同じサイズの配列生成
		im_mask = np.zeros((hight,width),np.uint8)
		# Trueの部分（背景）は白塗り
		im_mask[mask]=255
		# ゴマ塩ノイズ除去
		im_mask = cv2.medianBlur(im_mask,blur)

		# closing opening
		kernel = np.ones((8,8),np.uint8) # 10 10
	
		im_mask = cv2.morphologyEx(im_mask, cv2.MORPH_CLOSE, kernel)
		im_mask = cv2.morphologyEx(im_mask, cv2.MORPH_OPEN, kernel)
		# エッジ検出
		im_edge = cv2.Canny(im_mask,100,200)

		#path = "sample.jpg"
		#cv2.imwrite(path, im_mask)
		return im_mask, im_edge

	# hough変換での直線抽出から駒の先後判定
	def predict_sengo(self, ban_matrix, pos):
		img = ban_matrix.get_masu(pos[0], pos[1]).snippet_img
		ret = self._get_sengo_lines(img)
		prob = ret[1]
		
		if(self.log_obj):
			self.log_obj.log("predict sengo " + str(prob))
		return prob

	# 回転、平行移動した駒領域画像の配列取得、ただし get_koma_img を事前にしていること前提
	# ただし、回転はぼけるため、しないほうがいい
	def get_koma_img_vari(self, ban_matrix, x,y, trans, rot = 0):
		masu = ban_matrix.get_masu(x,y)
		cut_offset = masu.cut_offset
		start = [cut_offset[0] + trans[0], cut_offset[1] + trans[1]]
		goal = [start[0] + self.__cut_width, start[1] + self.__cut_height]
		if(start[0]<0):
			start[0] = 0
			goal[0] = self.__cut_width
		if(start[1] < 0):
			start[1] = 0
			goal[1] = self.__cut_height
		if(goal[0] > self.__row_ave):
			goal[0] = int(self.__row_ave)
			start[0] = int(self.__row_ave) - self.__cut_width
		if(goal[1] > self.__col_ave):
			goal[1] = int(self.__col_ave)
			start[1] = int(self.__col_ave) - self.__cut_height
		s_img = masu.snippet_img
		if(rot != 0):
			s_img = ndimage.rotate(s_img, rot)
		#rotated_img =  np.array(Image.fromarray(s_img).rotate(rot))

		cut_img = s_img[int(start[1]):int(goal[1]), int(start[0]):int(goal[0])]	
		
		return self.convert_arr_from_img(cut_img)



	# マスの画像から駒領域の画像の１次元配列を取得
	def get_koma_img(self, ban_matrix, x, y):
		masu = ban_matrix.get_masu(x,y)
		(cut_img, offset) = self.cut_along_edge_lines(masu.snippet_img)
		if(cut_img is None):
			return None
		masu.cut_offset = offset
		return self.convert_arr_from_img( cut_img)

	# 元画像からきりぬき
	def get_masu_color_img(self, ban_matrix, x, y):
		masu = ban_matrix.get_masu(x,y)
		cut = (masu.snippet_img_pos[0], masu.snippet_img_pos[1], masu.snippet_img_pos[0] + self.__row_ave, masu.snippet_img_pos[1] + self.__col_ave)
		return ban_matrix.img.crop(cut) 

	def get_koma_img_arr_threshold(self, ban_matrix, x, y):
		masu = ban_matrix.get_masu(x,y)
		(cut_img, offset) = self.cut_along_edge_lines(masu.snippet_img)
		if(cut_img is None):
			return None
		masu.cut_offset = offset
		c_img = self.get_masu_color_img(ban_matrix, x, y)
		im = np.array(c_img.convert('L'))
		im = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
		im = 255- im[ offset[1]:int(offset[1] + self.__cut_height), offset[0]:offset[0] + int(self.__cut_width)]
		return self.convert_arr_from_img(im)

	# 駒の端の直線に従って長方形に切り抜く
	def cut_along_edge_lines(self, img):
		[edge_lines,prob] = self.get_sengo_edge_lines(img)
		if(prob == 0):
			return (None, [0,0])

		piece_left = 0
		self.__cut_width = 24 # 30
		self.__cut_off = 10 # 4
		self.__cut_height = self.__col_ave - self.__cut_off
		
		if(edge_lines[0] != []):
			piece_left = self.__piece_left_from_left(edge_lines[0][0][0])
			if(piece_left == -1):
				if(edge_lines[1] != []):
					piece_left = self.__piece_left_from_right(edge_lines[1][0][0])
					if(piece_left == -1):
						piece_left = 0
				else:
					piece_left = self.__row_ave - self.__cut_width
		else:
			piece_left = self.__piece_left_from_right(edge_lines[1][0][0])
			if(piece_left == -1):
				piece_left = 0

		start = [int(piece_left), self.__cut_off/2]
		goal = [int(piece_left + self.__cut_width), int(self.__col_ave - self.__cut_off/2)]
		cut_img = img[start[1]:goal[1], start[0]:goal[0]]

		if(cut_img.shape[1] == 0):
			self.save_image(img, "temp/err.jpg")
			print "save err"
			#print cut_img.shape
			#print piece_left
		return (cut_img, start)

	# 駒領域の左端の計算式定義
	# ただしマスのりょいういきからはみだしてはいけない
	# 左右エッジのいずれからから求める式２つ
	def __piece_left_from_left(self, edge_pos):
		left = edge_pos + self.__cut_off
		if(left + self.__cut_width >= self.__row_ave):
			left = -1
		return left

	def __piece_left_from_right(self, edge_pos):
		left = edge_pos - self.__cut_width - self.__cut_off
		if(left <= 0):
			left = -1
		return left


	# 駒の両端の二線
	# 返り値 [edge_lines, prob]
	# edge_lines[0] = [[start],[goal]] #0が左側、　1が右側、startが駒の下側
	# prob: 先手が＋、　後手が -に
	def get_sengo_edge_lines(self, img):
		right_side = 0
		left_side = 0
		right_line= [0,0]
		left_line = [0,0]
		edge_lines = [[],[]]
		[lines, prob] = self._get_sengo_lines(img)

		if(prob ==0):
			return [edge_lines,0]

		for line in lines:
			
			if(prob > 0): # 上で交わる
				i_l = self.intersect_line_and_side_line(line[0], line[1], self.__col_ave)
			elif(prob < 0):
				i_l = self.intersect_line_and_side_line(line[0], line[1], 0)

			if(i_l < left_side ): # 横境界と中央より左側で交わる
				left_side = i_l
				left_line = line
			elif(right_side < i_l):
				right_side = i_l
				right_line = line

			if(prob > 0): # 上で交わる
				left_side_s = int(self.intersect_line_and_side_line(left_line[0], left_line[1], 0))
				right_side_s = int(self.intersect_line_and_side_line(right_line[0], right_line[1], 0))
						
			elif(prob < 0):
				left_side_s = int(self.intersect_line_and_side_line(left_line[0], left_line[1], self.__col_ave))
				right_side_s = int(self.intersect_line_and_side_line(right_line[0], right_line[1], self.__col_ave))

		if(left_side < 0):
			if(prob>0):
				edge_lines[0] = [[left_side + int(self.__row_ave/2), self.__col_ave], [left_side_s + int(self.__row_ave/2), 0]]
			else:
				edge_lines[0] = [[left_side + int(self.__row_ave/2), 0], [left_side_s + int(self.__row_ave/2), self.__col_ave]]
		if(right_side > 0):
			if(prob >0):
				edge_lines[1] = [[right_side + int(self.__row_ave/2), self.__col_ave], [right_side_s + int(self.__row_ave/2), 0]]
			else:
				edge_lines[1] = [[right_side + int(self.__row_ave/2), 0], [right_side_s + int(self.__row_ave/2), self.__col_ave]]

		return [edge_lines, prob]


	# 駒の両端の斜め線を取り出す（複数）
	# 返り値: [lines, prob]
	# lines[i] = [rho, theta] : マス区切りのローカル極座標 
	def _get_sengo_lines(self, img):
		#self.save_image(img, "temp/get_sengo_line.png")

		lines = cv2.HoughLines(img,1,np.pi/180,23) # 27

		prob = 0
		up_lines = [] # [[rho, theta], [] ..] # 上で交わる直線
		down_lines = []
		for line in lines:
			for rho,theta in line:
				if((np.pi * 45/180 < theta and  theta < np.pi *135 /180) or  theta == 0 or theta == np.pi/2 or theta == np.pi):
					continue
				l = self.intersect_line_and_center_line(rho, theta)
				if( self.__row_ave/2 < l and l < self.__row_ave * 4): # 下で交わる
					down_lines.append([rho, theta])
					prob -= 1

				elif(- 4*  self.__row_ave < l and l< -self.__row_ave/2):
					up_lines.append([rho, theta])
					prob += 1

		#self.draw_lines_from_polar(img, lines)
		#self.save_image(img, "temp/get_sengo_lines.jpg")

		if(prob == 0): # 先後判定できない
			lines = []
		elif(prob < 0):
			lines = down_lines
		else:
			lines = up_lines
		return [lines, prob]


	# 極座標表記の直線と縦の中心線との交点の中心からの位置
	def intersect_line_and_center_line(self, rho, theta):
		return int(rho/np.sin(theta) - self.__row_ave/2.0/np.tan(theta) - self.__col_ave/2.0)

	# 極座標表記の直線と、横の直線の交点位置
	def intersect_line_and_side_line(self, rho, theta, side):
		return int(-side*np.tan(theta) + rho/np.cos(theta) - self.__row_ave/2.0)

	# 画像ファイル保存
	# cv2 と PIL で配列の順序が違うのか,ＲＧＢが反転する
	def save_image(self, img, path=""):
		if(path == ""):
			path = "temp/save_image" + str(self.__save_image_num) + ".jpg"
			self.__save_image_num += 1
		Image.fromarray(img).save(path)
		#cv2.imwrite(path, img)

	def draw_board_position_list(self, im):
		for l in self.board_position_list:
			for ll in l:
				x = ll[0][0]
				y = ll[0][1]
				x2 = ll[1][0]
				y2 = ll[1][1]
				cv2.circle(im,(x,y),2,(0,0,255),3)
				self.draw_frame(im, (x,y), (x2, y2), wid = 1)
		return im


	# こまの先後を判定するサンプル
	def test_sample(self):
		import image_e
		#img = Image.open("server/capture.jpg")
		img = Image.open("./images/init_ban1.jpg")

		img = Image.fromarray(ndimage.rotate(img, -2))
	
		#img = Image.open("./init_ban2.jpg")
		img_ori = copy.copy(img)
		im = np.array(img.convert('L'))
		frame = cv2.Canny(im,70,170) # 10 170
		#frame = cv2.convertScaleAbs(cv2.Laplacian(im, cv2.CV_32F, 8))
		self.save_image(frame, "result0.jpg")

		im2 = np.array(frame)

		minLineLength = 4
		maxLineGap = 10
		lines = cv2.HoughLinesP(im2,1,np.pi/180,100,minLineLength,maxLineGap)
		#print lines
		arr_img = np.array(img)
		empty = np.zeros(im2.shape, dtype=np.uint8)
		#Image.fromarray(im2).show()
		for i in lines:
			for x1,y1,x2,y2 in i:
				cv2.line(arr_img,(x1,y1),(x2,y2),(0,255,0),1)
				cv2.line(empty,(x1,y1),(x2,y2),(255),1)
		#
		cv2.imwrite('result.jpg', empty)

		lines = cv2.HoughLines(empty,1,np.pi/180,90) # 180
		c_lines = []
		for i in xrange(len(lines)):
			c_lines.append(lines[i][0])

		self.draw_lines_from_polar(arr_img, c_lines)
		cv2.imwrite('result.jpg',arr_img)


	
		lines = cv2.HoughLines(im2,1,np.pi/180,150) # 180
		#lines = self._grouping_hough_lines(lines, 5)

		row_list = []
		col_list = []


		for line in lines:
			for rho,theta in line:
				if( rho < 0):
					line[0][1] -= np.pi
					line[0][0] = -line[0][0]
				if(abs(line[0][1]) < np.pi*20/180):
					col_list.append(line[0])
				elif(abs(line[0][1] - np.pi*90/180) < np.pi*20/180):
					row_list.append(line[0])

		
		grouping_rows = row_list
		grouping_cols = col_list
		grouping_rows = self._grouping_hough_lines(row_list, 5)
		grouping_cols = self._grouping_hough_lines(col_list, 5)

		#(grouping_cols, median) = self._get_board_lines_from_hough(col_list)
		#(grouping_rows, median) = self._get_board_lines_from_hough(row_list)

		arr_img = np.array(img_ori)

		self.draw_lines_from_polar(arr_img, grouping_cols)
		#self.draw_lines_from_polar(arr_img, col_list)
		self.draw_lines_from_polar(arr_img, grouping_rows)
		self.save_image(arr_img, "result2.jpg")
		
		print "result2 save"

		arr_img = np.array(img)

		board_position_list = self.get_board_position(im2)
		#print board_position_list
		self.board_position_list = board_position_list
		self.draw_board_position_list(arr_img)
		
		self.save_image(arr_img, "result3.jpg")


		img_list = self.devide_img_from_board_list(im2, board_position_list)


		arr_img = np.array(img)

		print self.__col_ave
		print self.__row_ave

		
		ban_matrix = image_e.BanMatrix(img_ori)

		self.set_masu_from_one_pic(ban_matrix)
		ban_matrix.set_init_placement()

		masu = ban_matrix.get_masu(1,1)
		(cut_img, offset) = self.cut_along_edge_lines(masu.snippet_img)
		print "cut img shape = " + str(cut_img.shape)
		
		snippet_img_arr = self.get_koma_img(ban_matrix, 1,1)
		snippet_img_arr2 = self.get_koma_img_vari(ban_matrix, 1,1, [0,0], 1)
		print np.where((snippet_img_arr2 - snippet_img_arr) != 0)
		#print np.where(snippet_img_arr != 0)
		roated_arr = self.get_img_from_arr(copy.copy(snippet_img_arr2), True)
		self.save_image(roated_arr*255, "temp/rotated2.jpg")

		self.save_image(cut_img, "temp/ori_img.jpg")
		
		self.flg = 0
		for x in range(9):
			print ("  " + str(x) + "=======")
			for y in range(9):
				self.flg += 1
				target = ban_matrix.get_koma(x,y)
				if(target == 0):
					continue
				s_img_pos = [board_position_list[0][0][0][0], board_position_list[0][0][0][1]]
				r_img = img_list[x][y]
				masu = square_space.Masu(x, y , r_img, 0, s_img_pos)
				img = r_img
				if(True):
					[edge_lines,prob] = self.get_sengo_edge_lines( img)
					if(prob == 0):
						continue

					masu_offset = board_position_list[x][y]

					piece_left = 0
					self.__cut_width = 30
					self.__cut_off = 4

					if(edge_lines[0] != []):
						piece_left = self.__piece_left_from_left(edge_lines[0][0][0])
						if(piece_left == -1):
							if(edge_lines[1] != []):
								piece_left = self.__piece_left_from_right(edge_lines[1][0][0])
								if(piece_left == -1):
									piece_left = 0
							else:
								piece_left = self.__row_ave - self.__cut_width
					else:
						piece_left = self.__piece_left_from_right(edge_lines[1][0][0])
						if(piece_left == -1):
							piece_left = 0

					for i in range(2):
						if(edge_lines[i] != []):
							start = (int(masu_offset[0][0] + edge_lines[i][0][0]), int(masu_offset[0][1] + edge_lines[i][0][1]))
							goal = (int(masu_offset[0][0] + edge_lines[i][1][0]), int(masu_offset[0][1] + edge_lines[i][1][1]))
							color = (255,0,0)
							cv2.line(arr_img,start,goal, color,1)
					
					start = [int(piece_left) + masu_offset[0][0], self.__cut_off/2 + masu_offset[0][1]]
					goal = [int(piece_left + self.__cut_width) + masu_offset[0][0], int(self.__col_ave - self.__cut_off/2)+ masu_offset[0][1]]
					self.draw_frame(arr_img, start, goal, wid = 1)

					print prob
					if(edge_lines[0] != [] and edge_lines[1]!= [] and edge_lines[1][0][0] - edge_lines[0][0][0] > self.__piece_width_ave):
						self.__piece_width_ave = edge_lines[1][0][0] - edge_lines[0][0][0]
		self.save_image(arr_img)
		print self.__piece_width_ave

	# hough変換を使用して盤の交点位置を取得
	def get_board_position(self, img):
		lines = cv2.HoughLines(img,1,np.pi/180,150)

		row_list = []
		col_list = []

		# 角度で縦と横分ける
		for line in lines:
			for rho,theta in line:
				if( rho < 0):
					line[0][1] -= np.pi
					line[0][0] = -line[0][0]
				if(abs(line[0][1]) < np.pi*20/180):
					col_list.append(line[0])
				elif(abs(line[0][1] - np.pi*90/180) < np.pi*20/180):
					row_list.append(line[0])

		(board_cols, median) = self._get_board_lines_from_hough(col_list)
		if(median == -1):
			return None
		elif(self.__row_ave == 0):
			self.__row_ave = median
			print "row_ ave: " + str(median)
		
		(board_rows, median) = self._get_board_lines_from_hough(row_list)
		if(median == -1):
			return None
		elif(self.__col_ave == 0):
			self.__col_ave = median
			print "col_ ave: " + str(median)


		board_position_list = []
		for row_line in board_rows:
			row_position_list = []
			for col_line in board_cols:
				intersect = self.calc_intersection(row_line, col_line)
				x = int(intersect[0] * math.cos(intersect[1]))
				y = int(intersect[0]*math.sin(intersect[1]))
				row_position_list.append([x,y])

			board_position_list.append(row_position_list)

		# ４すみの平均とって誤差を減らす
		minus_x = int(self.__row_ave/2)
		minus_y = int(self.__col_ave/2)
		if(self.__row_ave%2 == 0):
			plus_x = int(self.__row_ave/2)
		else:
			plus_x = int(self.__row_ave/2) + 1

		if(self.__col_ave%2 == 0):
			plus_y = int(self.__col_ave/2)
		else:
			plus_y = int(self.__col_ave/2) + 1

		average_board_position_l = []
		for x in range(9):
			average_row = []
			for y in range(9):
				center_pos_x = int((board_position_list[x][y][0] + board_position_list[x][y+1][0] + board_position_list[x+1][y][0] + board_position_list[x+1][y+1][0])/4)
				center_pos_y = int((board_position_list[x][y][1] + board_position_list[x][y+1][1] + board_position_list[x+1][y][1] + board_position_list[x+1][y+1][1])/4)
				x1 = center_pos_x - minus_x
				x2 = center_pos_x + plus_x
				y1 = center_pos_y - minus_y
				y2 = center_pos_y + plus_y
				average_row.append([[x1, y1],[x2, y2]])
				#average_row.append([board_position_list[x][y]])
			average_board_position_l.append(average_row)


		return average_board_position_l
		#return board_position_list
							#
	# hough lines をグルーピングする thresholdないのものが同じグループ
	def _grouping_hough_lines(self, hough_lines, threshold):
		sorted_lines = sorted(hough_lines,  key=lambda line: line[0])
		last_border = 0
		grouping_list = []
		theta_sum = 0
		rho_sum = 0

		center_line = sorted_lines[int(len(sorted_lines)/2)]
		if(abs(center_line[1]) < np.pi*10/180 ): #縦
			axis = 1
		else:
			axis = 0

		ref_line = [center_line[0], np.pi/2.0 * axis]

		for i in xrange(len(sorted_lines)):
			rho_sum += sorted_lines[i][0]
			theta_sum += sorted_lines[i][1]
			if(i != len(sorted_lines) -1):
				c1 = sorted_lines[i + 1][0] + self._get_line_grad_correction(sorted_lines[i+1],axis)
				c2 = sorted_lines[i ][0] + self._get_line_grad_correction(sorted_lines[i],axis)
				#print  self._get_line_grad_correction(sorted_lines[i+1],axis)
				
			# threthold 以上のものは別グループ
			if(i == len(sorted_lines)-1 or c1 - c2 > threshold):
				size = i +1 - last_border

				cut_lines = sorted_lines[last_border: i+1]
				grouping_list.append([rho_sum/size, theta_sum/size])
				rho_sum = 0
				theta_sum = 0
				last_border = i +1
		
		#return sorted_lines
		return grouping_list

	# 線の傾き分の補正
	def _get_line_grad_correction(self, line, axis):
		if(axis == 0):
			c = 200*(line[1] - np.pi/2.0)
		else:
			c = 200*(- line[1])
		return c

	# hough lineから盤の位置特定に必要な１０このライン抽出
	def _get_board_lines_from_hough(self, hough_lines):
		grouping_list = self._grouping_hough_lines(hough_lines, 5)
		size = len(grouping_list)
		max_distance = 0
		distance_list = []
		board_lines = []

		center_line = grouping_list[int(size/2)]
		if(abs(center_line[1]) < np.pi*10/180 ): #縦
			axis = 1
		else:
			axis = 0


		for i in xrange(len(grouping_list) -1):
			c1 = grouping_list[i + 1][0] + self._get_line_grad_correction(grouping_list[i+1],axis)
			c2 = grouping_list[i ][0] + self._get_line_grad_correction(grouping_list[i],axis)
			
			distace = c1 - c2
			distance_list.append(distace)
			
		median = sorted(distance_list)[int((size)/2)]

		start = 0
		while(not(len(board_lines) in[10]) and start < size):
			board_lines = [grouping_list[start]] # 1こめ
			crt_dist = 0
			for i in xrange(len(distance_list) -start):
				crt_dist += distance_list[i + start]
				if(abs(crt_dist - median) < 5): # median と値がほぼ一致なら正しい
					board_lines.append(grouping_list[i+start+1])
					crt_dist = 0

			start += 1
			

	
		if(len(board_lines) != 10):
			if(self.log_obj):
				self.log_obj.warning("lines number err" + str(len(board_lines)))
			else:
				print "get board lines from list err"
				print len(board_lines)
			return (board_lines, -1)

		return (board_lines , median)

	# line_list: 極座標表記のlineのlist
	def draw_lines_from_polar(self, img, line_list):
		_line_list = []
		if(len(line_list[0]) == 1):
			for line_packed in line_list:
				_line_list.append(line_packed[0])
		else:
			_line_list = line_list

		for rho,theta in _line_list:		
			a = np.cos(theta)
			b = np.sin(theta)
			x0 = a*rho
			y0 = b*rho
			x1 = int(x0 + 1000*(-b))
			y1 = int(y0 + 1000*(a))
			x2 = int(x0 - 1000*(-b))
			y2 = int(y0 - 1000*(a))
			color = (255,0,0)
			cv2.line(img,(x1,y1),(x2,y2), color,1)

	# line = [rho, theta]
	def calc_intersection(self, line1, line2):
		theta_dif = line1[1] - line2[1]
		theta_d = math.atan2(( - line2[0]/ line1[0] + math.cos(theta_dif)), math.sin(theta_dif))
		theta = line1[1] + theta_d
		rho = line1[0] / math.cos(theta_d)

		return [rho, theta]

	# 直交座標への変換
	def calc_xy_from_polar(self, polar):
		return [int(polar[0]*np.cos(polar[1])), int(polar[0]*np.sin(polar[1]))]







if False:
	from PIL import Image
	im = Image.open("sample105.jpg")

	im2 = Image.open("sample106.jpg")

	#KomaRecognition().get_tip_abspos(im, im2, [28, 144])
	rec = KomaRecognition()
	im3,  im_edge = rec.bg_diff(im, im2)
	rec.labeling(im_edge)


	if False:
		path = "sample.jpg"
		ban_region = (0,0, 100,200)
		im3 = im2.crop(ban_region)
		cv2.imwrite(path, np.array(im3))
		

if False:
	img = cv2.imread('sample105.jpg',0)
	img = cv2.medianBlur(img,5)
	cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

	circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
	circles = np.uint16(np.around(circles))
	for i in circles[0,:]:
	 # draw circles
	 cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
	 # draw center of circles
	 cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

	cv2.imshow('detected circles',cimg)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	cv2.imwrite('sample.jpg',cimg)



	
		 
if __name__ == '__main__':
	my_class = KomaRecognition()
	my_class.test_sample()