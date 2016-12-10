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
import WWW

import pickle

import threading
import sys
import datetime

col_ave = 0
row_ave = 0

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

class Masu:

	def __init__(self, x, y, s_img = [], _koma = 0, s_img_pos = [0,0]):
		self.set(x, y, s_img, _koma, s_img_pos)

	def set(self,  x, y, s_img = [], _koma = 0, s_img_pos = [0,0]):
		if(s_img == []):
			s_img = np.array([])
		self.x_pos = x
		self.y_pos = y
		self.snippet_img = s_img
		self.koma = _koma
		self.is_koma = not(_koma == 0) # 駒があるか
		self.snippet_img_pos = s_img_pos
		self.snippet_img_size = np.array(self.snippet_img).shape # snippet の元サイズ

		self.snippet_arr_size = 0
		self.snippet_img_arr = None

	def kihu_pos(self):
		return [9 - y, x + 1]

# 盤面管理
# ゲーム上のルールを管理する
# 認識処理は入らないようにする
class BanMatrix:

	def __init__(self, img = np.array([])):

		self.img = img
		self.__capture = [[],[]] # 持ち駒 # capture[0] 先手 駒一つにつきpushする
								# 配列にするのあまりイケてない気が.. どう管理するのがいい？少なくとも外部からは隠ぺいすべき
		self.masu_data = []
		for x in range(9):
			y_dat = []
			for y in range(9):
				masu = Masu(x, y)
				y_dat.append(masu)
			self.masu_data.append(y_dat)

	def set_koma(self, x, y , koma):
		self.masu_data[x][y].koma = koma
		self.masu_data[x][y].is_koma = False if koma == 0 else True

	def get_koma(self, x,y):
		if(self.valid_masu(x,y) == 0):
			return None
		return self.masu_data[x][y].koma

	# 持っている駒数 koma ==0 ですべての種類
	def capture_num(self, teban, koma = 0):
		num = 0
		if teban == 1:
			index = 0
		else:
			index = 1
		if koma == 0:
			num = len(self.__capture[index])
		else:
			num = self.__capture[index].count(koma)
		return num
	
	# 持ち駒追加
	def capture_add(self, teban, koma, num = 1):
		if teban == 1:
			index = 0
		else:
			index = 1
		for i in range(num):
			self.__capture[index].append(koma)

	# 持ち駒を使う（リストから削除）
	def capture_revive(self, teban, koma, num = 1):
		if teban == 1:
			index = 0
		else:
			index = 1
		for i in range(num):
			self.__capture[index].remove(koma)

	# 重複のない持ち駒種類リスト
	def capture_variety(self, teban):
		if teban == 1:
			index = 0
		else:
			index = 1
		return list(set(self.__capture[index])) # 重複をなくす

	def get_near_masu(self, x, y):
		near_list = []
		search_list = [[0,1], [1,0],[0,-1], [-1,0],[1,1],[1,-1],[-1,1],[-1,-1]]
		if(self.valid_masu(x,y) == 0):
			return near_list
		for search_pos in search_list:
			if(self.valid_masu(x + search_pos[0], y+search_pos[1])):
				near_list.append(search_pos)
		return near_list

	# 枠外対応
	def valid_masu(self, x, y):
		if(x<0 or x>8 or y<0 or y>8 ):
			return 0
		return 1

	def get_masu(self, x,y):
		if(self.valid_masu(x,y) == 0):
			return None
		return self.masu_data[x][y]

	def get_masu_color_img(self, x, y):
		masu = self.masu_data[x][y]
		#return self.img.crop((masu.snippet_img_pos[0], masu.snippet_img_pos[1], masu.snippet_img_pos[0] + masu.snippet_img_size[0], 
		#	masu.snippet_img_pos[1] + masu.snippet_img_size[1]))
		return self.img.crop((masu.snippet_img_pos[1], masu.snippet_img_pos[0], masu.snippet_img_pos[1] + masu.snippet_img_size[1],
		 masu.snippet_img_pos[0] + masu.snippet_img_size[0]))

	def print_ban(self, over_write = False):

		strl = ""
		strl +=  "\n 9  8  7  6  5  4  3  2  1 \n"
		strl +=  "---------------------------\n"
		for x in xrange(9):
			for y in xrange(9):
				koma = self.masu_data[x][y].koma
				if koma == 0 and self.masu_data[x][y].is_koma == True:
					koma = 1
				if(koma < 0):
					strl += str(int(koma))
				elif (koma > 0):
					strl += " " + str(int(koma))
				else:
					strl += "  "
				strl += " "
			strl += "|" + str(x+1) + "\n"
		if over_write:
			sys.stdout.write("\033[11\r%s" % strl)
			sys.stdout.flush()
		else:
			print strl



# 棋譜
# 打つ、左、とかもやりたい
# 盤面によらない各駒の性質（動きについても管理）
class Kihu:
	
	def __init__(self, teban, epoch, target, prev_pos, next_pos, revival = 0, promotion = 0):
		self.__promotion_list = [[HU,KYO,KEI,GIN,KAK,HIS], [TO,NKYO,NKEI,NGIN,UMA,RYU]]
		
		self.kihu_txt = ""
		self.target = 0 # koma
		self.prev_pos = []
		self.next_pos = []
		self.revival = 0 # 打つか
		self.promotion = 0

		self.teban = teban
		self.epoch = epoch
		self.prev_pos = prev_pos
		self.next_pos = next_pos
		self.target = target 
		self.revival = revival
		self.promotion = promotion

		now = datetime.datetime.now()
		self.create_time = now.strftime("%Y/%m/%d %H:%M:%S")

	# ここでは単なる動きの確認を行う
	def validation_check(self):
		valid = 1

		# 行き場のない駒
		koma = abs(self.target)
		if(koma in [HU, KEI, KYO]):
			if self.teban == 1 and self.next_pos[0] == 0:
				valid = 0
			elif self.teban == -1 and self.next_pos[0] == 8:
				valid = 0

		if(koma == KEI):
			if self.teban == 1 and self.next_pos[0] == 1:
				valid = 0
			elif self.teban == -1 and self.next_pos[0] == 7:
				valid = 0

		if valid == 0:
			return valid

		if self.revival:
			if self.promotion:
				valid = 0

		else:
			move = np.array([0,0])
			move[0] = self.next_pos[0] - self.prev_pos[0]
			move[1] = self.next_pos[1] - self.prev_pos[1]
			if self.teban < 0:
				move = -move
			if koma == 1 :
				if(not(move[0] == -1 and move[1] == 0)):
					valid = 0

			elif koma == 2:
				if(not(move[0] < 0 and move[1] == 0)):
					valid = 0

			elif koma == 3:
				if(not(abs(move[1]) == 1 and move[0] == -2 )):
					valid = 0

			elif koma == 4:
				if(not( (abs(move[0]) == 1 and abs(move[1]) == 1) or (move[0] == -1 and move[1] == 0))):
					valid = 0
			elif koma == 5:
				if(not( (abs(move[0]) == 1 and move[1] == 0) or (abs(move[1]) == 1 and move[0] == 0)  or (move[0] == -1 and abs(move[1])==1 )) ):
					valid = 0
			elif koma == 6:
				if(not(abs(move[0]) == abs(move[1]))):
					valid = 0
			elif koma == 7:
				if(move[0] * move[1] != 0 or (move[0] == 0 and move[1] == 0)):
					valid = 0
			elif koma == 8:
				if ((move[0] == 0 and move[1] == 0) or (abs(move[0]) > 1) or (abs(move[1] > 1))):
					valid = 0 
			
			if self.promotion: # なり確認 # ここでやる必要ある？
				valid = self.promotion_validation()

		return valid

	# 成ることができるか 既に成りごまの場合false
	def promotion_validation(self):
		valid = 0
		koma = abs(self.target)
		if(koma in self.__promotion_list[0]):
			if(self.teban == 1):
				if(not(self.prev_pos[0] > 2 and self.next_pos[0] > 2)):
					valid = 1
			else:
				if(not(self.prev_pos[0] < 6 and self.next_pos[0] < 6)):
					valid = 1
				
		return valid

	def get_txt(self, utf8 = False):
		txt = ""
		if(self.teban == 1):
			txt += "▲"
		else:
			txt += "△"

		txt += str(9 - self.next_pos[1])
		txt += str(self.next_pos[0] + 1)
		txt += self.get_koma_txt(self.target)
		if(self.promotion):
			txt += "成"

		txt += "\n"
		if utf8 :
			return txt.decode('utf-8')
		return txt

	def get_koma_txt(self, koma):
		koma = abs(koma)
		txt = ""
		if koma == HU:
			txt = "歩"
		elif koma == KYO:
			txt = "香"
		elif koma == KEI:
			txt = "桂"
		elif koma == GIN:
			txt = "銀"
		elif koma == KIN:
			txt = "金"
		elif koma == KAK:
			txt = "角"
		elif koma == HIS:
			txt = "飛"
		elif koma == OHO:
			txt = "王"
		elif koma == TO:
			txt = "と"
		elif koma == NKYO:
			txt = "杏"
		elif koma == NKEI:
			txt = "圭"
		elif koma == NGIN:
			txt = "全"
		elif koma == UMA:
			txt = "馬"
		elif koma == RYU:
			txt = "竜"
		return txt

	# promotion できるならpromoption した値が、できない場合、0が帰る
	def is_promotion(self, target = 0):
		if target == 0:
			target = self.target
		ret = 0
		target_abs = abs(target)
		print "target_abs" + str(target_abs)
		if(target_abs in self.__promotion_list[0]):
			index = self.__promotion_list[0].index(target_abs)
			ret = self.__promotion_list[1][index]
			if target < 0:
				ret = -ret
		return ret

	def is_cancel_promotion(self, target = 0):
		if target == 0:
			target = self.target
		ret = 0
		target_abs = abs(target)
		if(target_abs in self.__promotion_list[1]):
			index = self.__promotion_list[1].index(target_abs)
			ret = self.__promotion_list[0][index]
			if target < 0:
				ret = -ret
		return ret

	def set_promotion(self):
		
		self.promotion = 1

	def send_data(self, id):
		end_point = URL + "api/game/" + str(id) + "/game_record/add"
		is_promotion_num = 1 if self.promotion else 0
		
		if(self.prev_pos == []):
			position = "99" + str(self.next_pos[0]) + str(self.next_pos[1])
		else:
			position = str(self.prev_pos[0]) + str(self.prev_pos[1]) + str(self.next_pos[0]) + str(self.next_pos[1])
		
		post_data = {
			"epoch": self.epoch,
			"position": position,
			"target": self.target,
			"is_promotion": str(is_promotion_num),
			"revival" : str(self.revival),
			"kihu" : self.get_txt(),
			"create_time" : self.create_time 
		}

		print post_data
		print WWW.Post(end_point, post_data)



# 一連の局面進行を管理する
class PlayFlow:
	

	def __init__(self):

		self.crt_ban_matrix = None
		self.epoch = 0 # 手数
		self.kihus = []
		self.video_cap = None
		self.cap_thread = None
		self.start_flg = 0
		self.epoch = 0
		self.crt_ban_matrix = BanMatrix()
		self.teban = 1

		self.thred_time = 0.3

		self.crt_ban_matrix.set_koma(0,0, -KYO)
		self.crt_ban_matrix.set_koma(0,1, -KEI)
		self.crt_ban_matrix.set_koma(0,2, -GIN)
		self.crt_ban_matrix.set_koma(0,3, -KIN)
		self.crt_ban_matrix.set_koma(0,4, -OHO)
		self.crt_ban_matrix.set_koma(0,5, -KIN)
		self.crt_ban_matrix.set_koma(0,6, -GIN)
		self.crt_ban_matrix.set_koma(0,7, -KEI)
		self.crt_ban_matrix.set_koma(0,8, -KYO)

		self.crt_ban_matrix.set_koma(8,0, KYO)
		self.crt_ban_matrix.set_koma(8,1, KEI)
		self.crt_ban_matrix.set_koma(8,2, GIN)
		self.crt_ban_matrix.set_koma(8,3, KIN)
		self.crt_ban_matrix.set_koma(8,4, OHO)
		self.crt_ban_matrix.set_koma(8,5, KIN)
		self.crt_ban_matrix.set_koma(8,6, GIN)
		self.crt_ban_matrix.set_koma(8,7, KEI)
		self.crt_ban_matrix.set_koma(8,8, KYO)

		self.crt_ban_matrix.set_koma(1,1, -HIS)
		self.crt_ban_matrix.set_koma(1,7, -KAK)
		self.crt_ban_matrix.set_koma(7,1, KAK)
		self.crt_ban_matrix.set_koma(7,7, HIS)

		for i in range(9):
			self.crt_ban_matrix.set_koma(2,i, -HU)
			self.crt_ban_matrix.set_koma(6,i, HU)

		self.crt_ban_matrix.print_ban()

		now = datetime.datetime.now()
		crt_time = now.strftime("%Y/%m/%d %H:%M:%S")
		end_point = URL + "api/game/0/start" 
		ret = WWW.Post(end_point, {"start_time": crt_time})
		if not(ret['result']):
			print "Send Err !!!!!!"
			exit()

		self.game_id = ret['game_id']
		print "game_id = " + str(self.game_id)



	# スレッドを立てて一定時間ごとにカメラから読む
	def start_flow(self):
		self.video_cap = cv2.VideoCapture(0)
		self.start_flg = 1
		self.exec_new_image()


	def end_flow(self):
		# stop thread
		self.start_flg = 0
		# キャプチャの後始末と，ウィンドウをすべて消す
		self.video_cap.release()
		cv2.destroyAllWindows()

	# 最新画像を取り込んで処理
	def exec_new_image(self):
		if self.start_flg == 0:
			return 
		ret ,img = self.video_cap.read()
		# 画面に表示する
		cv2.imshow('frame0',img)
		
		# 1,盤のマス認識
		# 2,有無識別

		ban_matrix = BanMatrix(Image.fromarray(img))
		#self.crt_ban_matrix.print_ban()
		# 盤面セットに成功したら
		if set_masu_from_one_pic(ban_matrix):
			#ban_matrix.print_ban(over_write = False)

			# 前の局面と比較
			ret = self.judge_two_ban_matrix(self.crt_ban_matrix, ban_matrix)
			if not(ret['err'] == ""):
				print ret['err'].decode('utf-8') 
 			#print ret
			if ret['result'] == 1 and ret['changed'] == 1:
				# 手を進める
				self.flow_advance(ret['kihu'])
				print "epoch: " + str(self.epoch)
				print ret['kihu'].get_txt(utf8 = True)
				self.crt_ban_matrix.print_ban()

		else:
			print "board recognition err"

		self.cap_thread = threading.Timer(self.thred_time, self.exec_new_image) 
		self.cap_thread.start()

	# 手を進める
	# validation check はすましている前提
	def advance(self, kihu):

		# 駒打ちか
		if kihu.revival:
			self.crt_ban_matrix.capture_revive(kihu.teban, kihu.target)
			self.crt_ban_matrix.set_koma(kihu.next_pos[0], kihu.next_pos[1], kihu.target)
		
		else:
			self.crt_ban_matrix.set_koma(kihu.prev_pos[0], kihu.prev_pos[1], 0)
			# 駒取か
			next_koma = self.crt_ban_matrix.get_koma(kihu.next_pos[0], kihu.next_pos[1])
			if not(next_koma == 0):
				self.crt_ban_matrix.capture_add(kihu.teban, -next_koma) # 取得する駒は先後反転

			target = kihu.target
			if kihu.promotion:
				prom = kihu.is_promotion(target)
				target = prom

			self.crt_ban_matrix.set_koma(kihu.next_pos[0], kihu.next_pos[1], target)

	# game の局面が１つ進んだ
	def flow_advance(self, kihu):
		
		self.kihus.append(kihu)
		self.advance(kihu)
		kihu.send_data(self.game_id)
		self.epoch += 1
		self.teban = 1 - (self.epoch % 2) * 2 

	# 有無の変化は3種類
	# 1: +1 （打つ）
	# 2: +1 -1 (移動)
	# 3: -1  (駒取)  
	def judge_two_ban_matrix(self, ban1, ban2):
		dif_list = []
		result = 0
		changed = 0
		kihu = None
		err = ""

		for x in xrange(9):
			for y in xrange(9):
				#print str(ban1.get_masu(x,y).is_koma) + " " + str(ban2.get_masu(x,y).is_koma)
				if(ban1.get_masu(x,y).is_koma != ban2.get_masu(x,y).is_koma):
					dif_list.append([x, y])

		dif_num = len(dif_list)
	
		if(dif_num == 0):
			change = 0
			result = 1
		elif(dif_num == 1):
			if(ban2.get_masu(dif_list[0][0], dif_list[0][1]).is_koma ): # 駒打ち
				if(ban1.capture_num(self.teban) == 0):
					result = 0
				else:
					# todo 画像認識により、打ったコマを決定
					cap_variety = ban1.capture_variety(self.teban)
					if(len(cap_variety) == 1):
						kihu = Kihu(self.teban, self.epoch + 1, cap_variety[0], [], dif_list[0], revival = 1)
						result = 1
						changed = 1
						# 二歩チェック
						if abs(cap_variety[0]) == 1:
							for i in range(9):
								if ban1.get_koma(i, dif_list[0][1]) == cap_variety[0]:
									result = 0
									changed = 0
									err += "二歩です"
									break

					elif(len(cap_variety) > 1):
						# 複数打った駒候補があるときは識別する
						rec = self.recog_koma_in_list(ban1, cap_variety)
						kihu = Kihu(self.teban, self.epoch + 1, rec, [], dif_list[0], revival = 1)
						result = 1
						changed = 1
					else:
						err += "持ち駒にない駒は打てません"
			else: # 駒取
				# todo どこに移動したのか特定, TODO 成り判定も必要
				target = ban1.get_koma(dif_list[0][0], dif_list[0][1])
				movable_cap_list = self.get_movable_list(ban1, dif_list[0], capture = True) # 駒取のある移動地点のリストを取得
				max_probab = -1.0
			
				if(len(movable_cap_list) == 1):
					result = 1
					changed = 1
					kihu = Kihu(self.teban, self.epoch + 1, target, dif_list[0], movable_cap_list[0])
				
				elif(len(movable_cap_list) > 0):
					for cap_pos in movable_cap_list:
						probab = self.get_koma_probability(ban2, cap_pos, target, is_sengo_only = True) * self.teban # teban かけているのは、後手の場合、後手の駒である確からしさにするため
						if probab > max_probab:
							max_probab = probab
							next_pos = cap_pos
					
					kihu = Kihu(self.teban, self.epoch + 1, target, dif_list[0], next_pos)
					if kihu.promotion_validation(): # なっている可能性あり
						prom_koma = kihu.is_promotion(target)
						is_prom = self.recog_is_promotion(ban2, self.teban, next_pos, target)
						if is_prom:
							kihu.set_promotion()	

					result = 1
					changed = 1

		elif dif_num == 2: # 移動
			if ban1.get_masu(dif_list[0][0], dif_list[0][1]).is_koma :
				prev_pos = dif_list[0]
				next_pos = dif_list[1]
			else:
				prev_pos = dif_list[1]
				next_pos = dif_list[0]

			target = ban1.get_koma(prev_pos[0] , prev_pos[1])
			
			target_next_is_koma = ban2.get_masu(next_pos[0], next_pos[1]).is_koma
		

			_pass_check = self.pass_check(ban1, target, prev_pos, next_pos)
			if _pass_check == 0:
				err += "駒を通り越すことはできません"

			elif target * self.teban < 0:
				err += "手番が違います"

			elif  target_next_is_koma  : # target がnextで目標地点にいるか いらないかも
				kihu = Kihu(self.teban, self.epoch + 1, target, prev_pos, next_pos)
				if kihu.validation_check():
					if kihu.promotion_validation(): # なっている可能性あり
						if self.recog_is_promotion(ban2, self.teban, next_pos, target):
							print "prom!"
							kihu.set_promotion()
					result = 1
					changed = 1
				else:
					err += "そのように動けません"


		return {'result':result, 'changed' : changed, 'kihu': kihu, 'epoch': self.epoch, "teban": self.teban, "err":err}

	# 通過地点に駒がないか
	def pass_check(self, ban_matrix, target, prev_pos, next_pos):
		check = 1
		if not(abs(target) == KEI):
			x_dif = next_pos[0] - prev_pos[0] 
			y_dif = next_pos[1] - prev_pos[1] 
			dis = max(abs(x_dif), abs(y_dif))
			if dis > 1:
				xp = x_dif/dis
				yp = y_dif/dis
				for i in range(dis -1):
					pass_x = prev_pos[0] + (i+1) * xp
					pass_y = prev_pos[1] + (i+1) * yp
					if ban_matrix.get_masu(pass_x, pass_y).is_koma:
						check = 0
						break
		return check

	# 移動できる範囲のリストを取得　capture == 1 で駒取のみ取得
	def get_movable_list(self, ban_matrix, target_pos, capture = False):
		movable_list = []
		movable_cap_list = []
		target = ban_matrix.get_koma(target_pos[0], target_pos[1])
		abs_taget = abs(target)
		teban = 1 if target > 0 else -1
		dif_list = [] # 単体
		dif_vec_list = [] # 距離移動
		if abs_taget == HU:
			dif_list.append([1,0])
		elif abs_taget == KYO:
			dif_vec_list = [[1,0]]
		elif abs_taget == KEI:
			dif_list = [[2,1], [2,-1]]
		elif abs_taget == GIN:
			dif_list = [[1,1],[1,0], [1,-1],[-1,1],[-1,-1]]
		elif abs_taget in [KIN, NGIN, NKYO, NKEI]:
			dif_list = [[1,1], [1,0], [1,-1], [0,1],[0,-1], [-1,0]]
		elif abs_taget in [HIS, RYU]:
			dif_vec_list = [[1,0], [0,1], [-1,0], [0,-1]]
		elif abs_taget in [KAK, UMA]:
			dif_vec_list = [[1,1], [1,-1], [-1,1], [-1,-1]]
		elif abs_taget == OHO:
			dif_list = [[1,0], [1,1], [1,-1], [0,1], [0,-1], [-1,0], [-1,1], [-1,-1]]
		elif abs_taget == RYU:
			dif_list = [[1,1], [1,-1], [-1,1], [-1,-1]]
		elif abs_taget == UMA:
			dif_list = [[1,0], [0,1], [-1,0], [0,-1]]

		for dif in dif_list:
			tx = target_pos[0] - teban * dif[0]
			ty = target_pos[1] - teban * dif[1]
			koma = ban_matrix.get_koma(tx,ty)
			if(koma == 0):
				movable_list.append([tx, ty])
			elif not(koma == None) and koma * teban < 0:
				movable_cap_list.append([tx, ty])

		for dif_vec in dif_vec_list:
			for i in range(9):
				tx = target_pos[0] - teban * dif_vec[0] * (i+1)
				ty = target_pos[1] - teban * dif_vec[1] * (i+1)
				koma = ban_matrix.get_koma(tx,ty)
				if koma == 0:
					movable_list.append([tx, ty])
				elif koma == None: # 枠外
					break
				elif koma * teban > 0: # 味方
					break
				else:	# 敵駒
					movable_cap_list.append([tx, ty])
					break
		if capture :
			return movable_cap_list

		return movable_list + movable_cap_list

	# 識別むつかしそう　常にTrueでもいい？
	def recog_is_promotion(self, ban_matrix, teban, pos, target):
		kihu = Kihu(teban, self.epoch + 1, target, [], pos)
		prom_target = kihu.is_promotion()
		if prom_target == 0:
			return False
		koma_list = [target, prom_target]
		koma = self.recog_koma_in_list(ban_matrix, pos, koma_list)
		return koma == koma_list[1]

	# 駒候補の中から識別する 先後は別
	# 先後判定してからやりたい
	def recog_koma_in_list(self, ban_matrix, pos, koma_list):
		probab_list = []
		for index ,koma in enumerate(koma_list):
			probab_list.append(self.get_koma_probability(ban_matrix, pos, koma))
		index =  probab_list.index(max(probab_list))
		return koma_list[index]

	# 単体識別 指定した駒であるか 確からしさの値を返す
	def get_koma_probability(self, ban_matrix, pos, koma, is_sengo_only = False):
		koma_recognition = KomaRecognition(ban_matrix, pos)
		if is_sengo_only: # 先後のみ判定（先手なら１）
			prob = koma_recognition.get_sengo_probability()
		else:
			prob = koma_recognition.get_probability(koma)
		print pos
		print "prob = " + str(prob)
		return prob

# PlayFlow から分離して駒単体の認識を記述する
# Todo 画像処理のうち、駒、盤にかかわるもの（より汎用的なものは別クラスで）はこちらに組み込みたい
class KomaRecognition:
	
	# @class member
	SengoLearnFile = "sengo_recog.txt"
	SengoLearnW = None
	KomaLearnFile = "koma%d_recog.txt"
	KomaLearnW = []
	
	def __init__(self, ban_matrix, pos):
		
		self.ban_matrix = ban_matrix
		self.pos = pos
		self.taregt = ban_matrix.get_koma(pos[0], pos[1])
		if KomaRecognition.KomaLearnW == []:
			for i in range(14): # magic number
				KomaRecognition.KomaLearnW.append(None)


	def get_probability(self, koma):
		koma_abs = abs(koma)
		if KomaRecognition.KomaLearnW[koma_abs] == None:
			f = open(KomaRecognition.KomaLearnFile % (koma_abs))
			KomaRecognition.KomaLearnW[koma_abs] = pickle.load(f)
		is_inv = False
		if koma < 0:
			isinv = True

		return self.__get_probability(KomaRecognition.KomaLearnW[koma_abs], is_inv)

	def get_sengo_probability(self):
		if KomaRecognition.SengoLearnW == None:
			f = open(KomaRecognition.SengoLearnFile)
			KomaRecognition.SengoLearnW = pickle.load(f)
		return self.__get_probability(KomaRecognition.SengoLearnW)

	def __get_probability(self, learn_data, is_inv = False):
		arr = self.ban_matrix.get_masu(self.pos[0], self.pos[1]).snippet_img_arr
		if is_inv:
			arr = inverse_img(arr)
		ret =  DNN.recog(learn_data, arr, [0])
		return ret['result']

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

def recog_one_ban(img):
	
	# 1,盤のマス認識
	# 2,有無識別
	ban_matrix = BanMatrix(img)
	set_masu_from_one_pic(ban_matrix)
	#ban_matrix.print_ban()
	# 2, 先後識別
	for x in xrange(9):
 		for y in xrange(9):
			index = x*9 + y
			if ban_matrix.masu_data[x][y].is_koma == False:
				continue
			# 近傍空きマス取得
			near_list = ban_matrix.get_near_masu(x,y)
			empty_cnt = 0
			color_sum  = np.array([0.0,0.0,0.0])
			for pos in near_list:
				if(ban_matrix.masu_data[pos[0]][pos[1]].is_koma == False):
					cim = np.array(ban_matrix.get_masu_color_img(pos[0], pos[1]))
					size = ban_matrix.masu_data[pos[0]][pos[1]].snippet_img_size
					trim = 3
					cim2 = cim[trim:size[0] - trim, trim: size[1]-trim, :]
					color_sum += np.array([cim2[:, : ,0].mean(), cim2[:, : ,1].mean(), cim2[:, : ,2].mean()])
				empty_cnt += 1
			# 空マスの平均カラー取得
			if(empty_cnt > 0):
				color_mean = color_sum/empty_cnt
				cim = np.array(ban_matrix.get_masu_color_img(x, y))
				dev_img = np.zeros([cim.shape[0], cim.shape[1]])
				for x1 in xrange(cim.shape[0]):
					for y1 in xrange(cim.shape[1]):
						if(np.linalg.norm(cim[x1][y1] - color_mean) < 100):   
						 #if(np.linalg.norm(color_HSV(cim[x1][y1])[2] - color_HSV(color_mean)[2]) < 150):      # 100 以下　maxpooling での学習性能は？
						 	dev_img[x1][y1] = 255
						
				if(y == 1 ):
					Image.fromarray(dev_img).show()	
					
					#dev_img = DNN.max_pooling(dev_img)
					#Image.fromarray(dev_img).show()	 	

def color_HSV(color):
	max_val = max(color)
	min_val = min(color)
	h = 0
	s = 0
	v = 0
	if max_val == color[0]:
		h = 60 * (color[1] - color[2])/(max_val - min_val)
	elif max_val == color[1]:
		h = 60*(color[2] - color[0])/(max_val - min_val) + 120
	else:
		h = 60*(color[0] - color[1])/(max_val - min_val) + 240

	v = max_val
	s = (max_val - min_val)/max_val
	return np.array([h,s,v])

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
	print " 9  8  7  6  5  4  3  2  1 0"
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
	im = Image.open("init_ban6.jpg")
 	dat = make_data_from_one_pic(im, ban_data)
 	for x in xrange(9):
 		for y in xrange(9):
 			index = x*9 + y
 			if(is_koma(dat['x_data'][index])):
	 			
	 			ret = DNN.recog(learn_data, dat['x_data'][index], [dat['y_data'][index]])
	 			# 閾値ぎりぎりなのは反転して確かめ
	 			if(abs(ret['result'] -0.5) < 0.15 ):
	 				ret2 = DNN.recog(learn_data, dat['x_data'][index][::-1], [dat['y_data'][index]])
	 				ret['pred'] = 1 if ret['result'] - ret2['result'] > 0.0 else  0 
	 				#if(abs(ret['result'] -0.5) < abs(ret2['result'] -0.5)):
	 				#	ret['pred'] = 1 - ret2['pred']
	 			ban_data[x][y] = ret['pred']
	 			if(ban_data[x][y]  == 0):
	 				ban_data[x][y] = -1
	 		else:
	 			ban_data[x][y] = 0
 	#print_ban(ban_data)
	return

def set_masu_from_one_pic(ban_matrix):
	img = ban_matrix.img
	im = np.array(img.convert('L'))
	frame = cv2.Canny(im,80,180) # 10 170
	#frame = cv2.convertScaleAbs(cv2.Laplacian(im, cv2.CV_32F, 8))

	im2 = np.array(frame)
	#Image.fromarray(im2).show()
	line_list = get_line_list(im2)
	if line_list == []:
		return 0
	img_list = devide_img(im2, line_list)
	#Image.fromarray(add_line(im2, line_list)).show()

	for x in range(9):
		for y in range(9):
			s_img_pos = [line_list[1][x], line_list[0][y]]
			r_img = img_list[x][y]
			masu = Masu(x, y , r_img, 0, s_img_pos)
			img = max_pooling(r_img)
			#if(x==0 and y==0):
				#Image.fromarray(img).show()
			masu.snippet_img_arr = convert_arr_from_img(img)/255
			masu.is_koma = is_koma(masu.snippet_img_arr)
			ban_matrix.masu_data[x][y] = masu
	return 1



# ban_data : 教師データ
# learn_masu : 学習対象マス
# learn_koma : 学習する駒
def make_data_from_one_pic(img, ban_data = [], learn_masu = [], learn_koma = 0):
	if(ban_data == []):
		ban_data = np.zeros([9,9])

	if(learn_masu == []):
		for i in range(9):
			for j in range(9):
				learn_masu.append([i,j])

	im = np.array(img.convert('L'))
	frame = cv2.Canny(im,80,180)
	#frame = cv2.convertScaleAbs(cv2.Laplacian(im, cv2.CV_32F, 8))

	im2 = np.array(frame)
	#Image.fromarray(im2).show()
	line_list = get_line_list(im2)
	img_list = devide_img(im2, line_list)
	
	x_data = []
	y_data = []
	raw_img = []
	for masu in learn_masu:
		col = masu[0]
		row = masu[1]
		koma = ban_data[col][row]
		if(koma < 0):
			r_img = inverse_img(img_list[col][row])
		else:
			r_img = img_list[col][row]

		img = max_pooling(r_img)
		x_data.append(convert_arr_from_img(img)/255)
		raw_img.append(r_img)
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
	return {'x_data': x_data, 'y_data': y_data, 'raw_img': raw_img}

def is_koma(img_arr):
	global col_ave, row_ave
	trim_size = 4
	img_arr2 = copy.copy(img_arr.reshape([col_ave/2, row_ave/2]))
	img_arr2 = img_arr2[trim_size:col_ave/2 - trim_size, trim_size:row_ave/2 - trim_size]
	#print np.mean(img_arr2)
	if(np.mean(img_arr2) < 0.3):
		return False
	return True

def make_data():
	HU = 1
	KYO = 2
	KEI = 3
	GIN = 4
	KIN = 5
	KAK = 6
	HIS = 7
	OHO = 8
	#ban_data = ban_init()
	ban_data = np.zeros([9,9])
	for i in  range(9):
		for j in range(9):
			if(abs(i - j)%4 == 0):
				ban_data[i][j] = 1
			if(abs(i - j)%4 == 2):
				ban_data[i][j] = -1
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
			for j in range(9):
				if(abs(i -j) % 2 == 0):
					learn_masu.append([i,j])
	

	cnt = 0
	for i in range(file_num):
		path = "init_ban" + str(i + 101) + ".jpg"
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
#learn_data = make_data()
#chain_recog.learn(learn_data['x_data'], learn_data['y_data'])

#learn_data =  DNN.learn_m(learn_data['x_data'], learn_data['y_data'])
#f = open('learn_data.txt', 'w')
#pickle.dump(learn_data, f)
#f.close()
#f = open('learn_data.txt')
#learn_data = pickle.load(f)

#ban_recog(learn_data)

#im = Image.open("init_ban1.jpg")
#recog_one_ban(im)


flow = PlayFlow()
flow.start_flow()
while(True):
	# キーボード入力
	key = cv2.waitKey(1) & 0xFF

	if key == 27 or key == ord('q'): # esc
		flow.end_flow()
		break


print "successfully ended"




