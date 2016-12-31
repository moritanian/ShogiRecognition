#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
import glob
import os.path
#from scipy import ndimage
import cv2
import copy
#import matplotlib.pyplot as plt
#from scipy import dot, roll

#import chain_recog
import DNN
import apery_call
import WWW
import koma_recognition
import square_space
import wroom_master

import pickle

import threading
import sys
import datetime
import time
import inspect

from server import main_sock 

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


# 盤面管理
# ゲーム上のルールを管理する
# 認識処理は入らないようにする
class BanMatrix:

	def __init__(self, img = np.array([]), log_obj = None):

		self.log_obj = log_obj

		self.img = img
		self.edge_abspos = []
		self.__capture = [[],[]] # 持ち駒 # capture[0] 先手 駒一つにつきpushする
								# 配列にするのあまりイケてない気が.. どう管理するのがいい？少なくとも外部からは隠ぺいすべき
		self.masu_data = []
		for x in range(9):
			y_dat = []
			for y in range(9):
				masu = square_space.Masu(x, y)
				y_dat.append(masu)
			self.masu_data.append(y_dat)

	def set_init_placement(self):
		self.set_koma(0,0, -KYO)
		self.set_koma(0,1, -KEI)
		self.set_koma(0,2, -GIN)
		self.set_koma(0,3, -KIN)
		self.set_koma(0,4, -OHO)
		self.set_koma(0,5, -KIN)
		self.set_koma(0,6, -GIN)
		self.set_koma(0,7, -KEI)
		self.set_koma(0,8, -KYO)

		self.set_koma(8,0, KYO)
		self.set_koma(8,1, KEI)
		self.set_koma(8,2, GIN)
		self.set_koma(8,3, KIN)
		self.set_koma(8,4, OHO)
		self.set_koma(8,5, KIN)
		self.set_koma(8,6, GIN)
		self.set_koma(8,7, KEI)
		self.set_koma(8,8, KYO)

		self.set_koma(1,1, -HIS)
		self.set_koma(1,7, -KAK)
		self.set_koma(7,1, KAK)
		self.set_koma(7,7, HIS)


		for i in range(9):
			self.set_koma(2,i, -HU)
			self.set_koma(6,i, HU)


	def set_koma(self, x, y , koma):
		self.masu_data[x][y].koma = koma
		self.masu_data[x][y].is_koma = False if koma == 0 else True

	def get_koma(self, x,y):
		if(self.valid_masu(x,y) == 0):
			return None
		return self.masu_data[x][y].koma

	def get_snippet_img(self, x, y):
		if(self.valid_masu(x,y) == 0):
			return None
		return self.masu_data[x][y].snippet_img


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
			#print strl
			self.log_obj.log(strl)


# 棋譜
# 打つ、左、とかもやりたい
# 盤面によらない各駒の性質（動きについても管理）
class Kihu:
	
	# @static member
	__usi_targets_first = ['P', 'L', 'N', 'S', 'G', 'B', 'R', 'K', '+P', '+L', '+N', '+S', '+B', '+R']
	__usi_targets_second = ['p', 'l', 'n', 's', 'g', 'b', 'r', 'k', '+p', '+l', '+n', '+s', '+b', '+r']

	def __init__(self, teban = 0, epoch=0, target=0, prev_pos=[], next_pos=[], revival = 0, promotion = 0):
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

		self.is_pointed_good = False

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

	# 音声合成用テキスト
	def get_voice_txt(self, teban_voice = True):
		txt = ""
		if(self.teban == 1):
			txt += "先手"
		else:
			txt += "後手"

		txt += " "
		txt += str(9 - self.next_pos[1])
		txt += " "
		txt += str(self.next_pos[0] + 1)
		txt += " "
		txt += self.get_koma_voice_txt(self.target)
		if(self.promotion):
			txt += "なる"
		txt += " "
		return txt

	def get_usi(self): #usi形式で取得
		if(self.revival):
			usi_t = self.__usi_targets_first[abs(self.target) - 1]
			usi = usi_t + "*" + self.__usi_pos(self.next_pos)
		else:
			usi = self.__usi_pos(self.prev_pos) + self.__usi_pos(self.next_pos)
			if(self.promotion != 0):
				usi += "+"
		return usi

	def set_from_usi(self, usi):
		if(usi[0].isdigit()): # 移動
			self.prev_pos = self.__pos_from_usi(usi[0:2])
			self.next_pos = self.__pos_from_usi(usi[2:4])
			if(len(usi) == 5 and usi[4] == "+"):
				self.promotion = 1

		else: # 駒うち 
			if(usi[1] != '*'):
				sys.stderr.write('input usi was invalid. revival err')
				exit()
			self.next_pos = self.__pos_from_usi(usi[2:4])
			usi_t = usi[0]
			if(usi_t in self.__usi_targets_first):
				self.target = (self.__usi_targets_first.index(usi_t) + 1) * self.teban
				print "set from usi tartget"
				print self.target
			else:
				sys.stderr.write('input usi was invalid. target err')
				exit()
			self.revival = 1
		return self

	def __usi_pos(self, pos):
		return str(9-pos[1]) + chr(97 + pos[0])

	def __pos_from_usi(self, usi):
		print usi
		return [ord(usi[1]) - 97 , 9 - int(usi[0]) ]


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

	def get_koma_voice_txt(self, koma):
		koma = abs(koma)
		txt = ""
		if koma == HU:
			txt = "ふ-"
		elif koma == KYO:
			txt = "きょう"
		elif koma == KEI:
			txt = "けい"
		elif koma == GIN:
			txt = "ぎん"
		elif koma == KIN:
			txt = "きん"
		elif koma == KAK:
			txt = "かく"
		elif koma == HIS:
			txt = "ひ"
		elif koma == OHO:
			txt = "おう"
		elif koma == TO:
			txt = "と"
		elif koma == NKYO:
			txt = "なりきょう"
		elif koma == NKEI:
			txt = "なりけい"
		elif koma == NGIN:
			txt = "なりぎん"
		elif koma == UMA:
			txt = "うま"
		elif koma == RYU:
			txt = "りゅう"
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

		global log_obj 
		log_obj.log(str(post_data))
		ret = WWW.Post(end_point, post_data)
		if(log_obj.main_socket()):
			status = 0
			if(ret):
				status = 1
			log_obj.main_socket().set_info_status(status, node_id = 2)
		
		if(ret):
			log_obj.log(ret)
		else:
			log_obj.warning("server post err")



# 一連の局面進行を管理する
class PlayFlow:
	

	def __init__(self, log_obj = None):

		self.log_obj = log_obj
		self.is_apery_assist = False


		self.crt_ban_matrix = None
		self.epoch = 0 # 手数
		self.kihus = []
		self.video_cap = None
		self.cap_thread = None
		self.start_flg = 0
		self.epoch = 0
		self.crt_ban_matrix = BanMatrix(log_obj = self.log_obj)
		self.teban = 1

		self.thred_time = 0.1

		self.crt_ban_matrix.set_init_placement()

		self.crt_ban_matrix.print_ban()

		now = datetime.datetime.now()
		crt_time = now.strftime("%Y/%m/%d %H:%M:%S")
		end_point = URL + "api/game/0/start" 
		ret = WWW.Post(end_point, {"start_time": crt_time})
		if ret == None: # サーバアクセスできない
			self.game_id = 0 # デフォルト0にする

		elif not(ret['result']):
			print "Send Err !!!!!!"
			exit()
		else:
			self.game_id = ret['game_id']
		print "game_id = " + str(self.game_id)

		if(self.log_obj.main_socket()):
			status = 0
			if(ret):
				status = 1
			log_obj.main_socket().set_info_status(status, node_id = 2)

		self.koma_recognition = koma_recognition.KomaRecognition(is_learning = False,  log_obj = self.log_obj.copy_obj(12))

		self.is_learn = False # 機械学習するか

		self.__capture_path = "server/capture.jpg"
		self.__capture_count = 18
		self.newest_ban_matrix = None
		self.advanced_img = None



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
		# apery
		if self.is_apery_assist:
			self.apery.end_proc()

	# 最新画像を取り込んで処理 スレッドとして動作
	def exec_new_image(self):
		#self.start_time = time.time()
	
		if self.start_flg == 0:
			return 
		ret ,img = self.video_cap.read()
		# 画面に表示する
		cv2.imshow('frame0',img)
		
		# 1,盤のマス認識
		# 2,有無識別

		ban_matrix = BanMatrix(Image.fromarray(img), log_obj = self.log_obj)
		#self.crt_ban_matrix.print_ban()
		# 盤面セットに成功したら
		if self.koma_recognition.set_masu_from_one_pic(ban_matrix):
			#ban_matrix.print_ban(over_write = False)
			self.newest_ban_matrix = ban_matrix


			if(self.advanced_img):
				ans = self.koma_recognition.judge_img_diff(self.advanced_img, self.newest_ban_matrix)
			if(not(self.advanced_img) or ans["result"]):

				# 前の局面と比較
				ret = self.judge_two_ban_matrix(self.crt_ban_matrix, ban_matrix)
				if not(ret['err'] == ""):
					print ret['err'].decode('utf-8') 

				else:
					self.__capture_count += 1
					if(self.__capture_count == 20):

						#self.__capture_count = 0
						ban_matrix.img.save(self.__capture_path)
						self.advanced_img = ban_matrix.img

	 			#print ret
				if ret['result'] == 1 and ret['changed'] == 1:
					# 手を進める
					ban_matrix.img.save(self.__capture_path)
					self.advanced_img = ban_matrix.img

					self.__capture_count = 0

					self.flow_advance(ret['kihu'])
					self.log_obj.log( "epoch: " + str(self.epoch))
					self.log_obj.log(ret['kihu'].get_txt(utf8 = True))
					self.crt_ban_matrix.print_ban()

					if self.is_learn:
						self.koma_recognition.set_from_banmatrix(ban_matrix, self.crt_ban_matrix)

			

		else:
			self.log_obj.log("board recognition err")

		#elapsed_time = time.time() - self.start_time
		#print ("elapsed_time:{0}".format(elapsed_time)) + "[sec]"

		self.cap_thread = threading.Timer(self.thred_time, self.exec_new_image) 
		self.cap_thread.start()


	# 手を進める(crt_ban_matrix のみの更新) 
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
		if(self.is_apery_assist):
			print "flow advance move_board"
			self.apery.move_board(kihu.get_usi())

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
						rec = self.recog_koma_in_list(ban1, dif_list[0], cap_variety)
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
					next_pos = movable_cap_list[0]
					kihu = Kihu(self.teban, self.epoch + 1, target, dif_list[0], next_pos)
				
				elif(len(movable_cap_list) > 0):
					for cap_pos in movable_cap_list:
						probab = self.get_koma_probability(ban2, cap_pos, target, is_sengo_only = True) * self.teban # teban かけているのは、後手の場合、後手の駒である確からしさにするため
						if probab > max_probab:
							max_probab = probab
							next_pos = cap_pos
					
					kihu = Kihu(self.teban, self.epoch + 1, target, dif_list[0], next_pos)

				if(len(movable_cap_list) > 0):
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
		prob = self.get_koma_probability(ban_matrix, pos, target)
		self.log_obj.log("recog is prom" + str(prob))
		return prob < 0.5
		#koma_list = [target, prom_target]
		#koma = self.recog_koma_in_list(ban_matrix, pos, koma_list)
		#return koma == koma_list[1]

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
		if is_sengo_only: # 先後のみ判定（先手なら１）
			prob = self.koma_recognition.get_sengo_probability(ban_matrix, pos)
		else:
			prob = self.koma_recognition.get_probability(ban_matrix, pos, koma)
		print pos
		print "prob = " + str(prob)
		return prob

	def apery_assist(self, apery):
		global main_socket
		self.apery = apery
		self.is_apery_assist = True
		# 既に進んでいる場合、局面をaperyオブジェクトに渡す
		for kihu in self.kihus:
			self.apery.move_board(kihu.get_usi())
		if(self.log_obj and self.log_obj.main_socket()):
			self.log_obj.main_socket().set_info_status(1, node_id = 8)

	def get_apery_ans(self, by_wroom = False):
		if(not(self.apery)):
			self.log_obj.log("start apery assist", 8)
			apery = apery_call.AperyCall()
			self.apery_assist(apery)
		ans = self.apery.get_answer()
		kihu = Kihu(teban = self.teban).set_from_usi(ans[0])
		# usiでの棋譜データは移動の場合、targetを明記していないため、局面から取得する
		if(kihu.revival == 0):
			kihu.target = self.crt_ban_matrix.get_koma(kihu.prev_pos[0], kihu.prev_pos[1])
		
		if(by_wroom): # deviceのさす位置を推定し、あっているか判定
			pointed_pos = self.koma_recognition.get_pointed_pos(self.advanced_img, self.newest_ban_matrix)
			if(pointed_pos):
				self.log_obj.log("your point is " + str(pointed_pos[0]) + str(pointed_pos[1]) ,3)
				print kihu.next_pos
				kihu.is_pointed_good = (pointed_pos[0] == kihu.next_pos[0] and pointed_pos[1] == kihu.next_pos[1])
			else:
				self.log_obj.log("your point is not found!!" ,3)

				kihu.is_pointed_good = False

			self.log_obj.log("get_apery_ans by_room " +  str(kihu.is_pointed_good), 3)
		
		return kihu





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

# 独立に先後学習する (できるだけ既存のクラス、メソッドをしよう)
def sengo_test():
	komaRecognition = koma_recognition.KomaRecognition(is_learning = False)


	m_files = range(3)
	t_files = [3]
	learnW = None

	l_data_list = []
	t_data_list = []

	for i in m_files:
		im = Image.open("images/init_ban" + str(i) + ".jpg")
		ban_matrix = BanMatrix(im)
		
		ret = komaRecognition.set_masu_from_one_pic(ban_matrix)
		ban_matrix.set_init_placement()

		if(ret):
			for x in range(9):
				for y in range(9):
					target = ban_matrix.get_koma(x,y)
					abs_target = abs(target)
					l_data = []
					if(target > 0):
						l_data.append(1)
						l_data.append(ban_matrix.get_masu(x,y).snippet_img_arr)
						l_data_list.append(l_data)

					elif(target < 0):
						l_data.append(0)
						l_data.append(ban_matrix.get_masu(x,y).snippet_img_arr)
						#l_data.append(komaRecognition.inverse_img_arr(ban_matrix.get_masu(x,y).snippet_img_arr))
						l_data_list.append(l_data)
					
		else:
			print "waring set masu img"

	for i in t_files:
		im = Image.open("images/init_ban" + str(i) + ".jpg")
		ban_matrix = BanMatrix(im)
		ret = komaRecognition.set_masu_from_one_pic(ban_matrix)
		ban_matrix.set_init_placement()
		if(ret):
			for x in range(9):
				for y in range(9):
					target = ban_matrix.get_koma(x,y)
					abs_target = abs(target)
					t_data = []
					if(target > 0):
						t_data.append(1)
						t_data.append(ban_matrix.get_masu(x,y).snippet_img_arr)
						t_data_list.append(t_data)
						
					elif(target < 0):
						t_data.append(0)
						t_data.append(ban_matrix.get_masu(x,y).snippet_img_arr)
						#t_data.append(komaRecognition.inverse_img_arr(ban_matrix.get_masu(x,y).snippet_img_arr))
						t_data_list.append(t_data)

					
		else:
			print "waring set masu img"

	max_epoch = 20
	for i in xrange(max_epoch):
		err_ave = 0.0
		for l_data in l_data_list:
			(err, learnW) = DNN.batch_learn(learnW, l_data[1], l_data[0])	
			err_ave += err
			#print err
		err_ave /= len(l_data_list)
		print "       err= " + str(err_ave)
	# test
	success = 0
	index = 0
	for t_data in t_data_list:
		ret = DNN.recog(learnW, t_data[1], [t_data[0]])
		if ret["success"] == 1:
			success += 1
		index += 1
	# result
	print str(success) + "/" + str(index)

# 独立に学習する (できるだけ既存のクラス、メソッドをしよう)
# 歩の場合、6files 係数0.2~0.3で9割が限界？
def koma_test(koma = 4):
	komaRecognition = koma_recognition.KomaRecognition(is_learning = False)
	m_files = range(18)
	t_files = [18,19]
	learnW = None

	l_data_list = []
	t_data_list = []

	for i in m_files:
		im = Image.open("images/init_ban" + str(i) + ".jpg")
		ban_matrix = BanMatrix(im)
		
		ret = komaRecognition.set_masu_from_one_pic(ban_matrix)
		ban_matrix.set_init_placement()
		xs = [0,6,7,8]
		ys = [0,2,6,8]

		if(ret):
			for x in xs:
				for y in ys:
					target = ban_matrix.get_koma(x,y)
					if(target ==0 or ban_matrix.get_masu(x,y).snippet_img_arr is None):
						continue
					abs_target = abs(target)
					l_data = []
					if(target > 0):
						img = ban_matrix.get_masu(x,y).snippet_img_arr
					else:
						img = komaRecognition.inverse_img_arr(ban_matrix.get_masu(x,y).snippet_img_arr)

					if(img.shape[0] == 0):
						print "aaa"
						print x
						print y

					if(abs_target == koma):
						l_data.append(1)
						l_data.append(img)
						l_data_list.append(l_data)

					else:
						l_data.append(0)
						l_data.append(img)
						l_data_list.append(l_data)
			print img.shape
					
		else:
			print "waring set masu img"

	for i in t_files:
		im = Image.open("images/init_ban" + str(i) + ".jpg")
		ban_matrix = BanMatrix(im)
		ret = komaRecognition.set_masu_from_one_pic(ban_matrix)
		ban_matrix.set_init_placement()
		if(ret):
			for x in xs:
				for y in ys:
					target = ban_matrix.get_koma(x,y)
					if(target ==0 or ban_matrix.get_masu(x,y).snippet_img_arr is None):
						continue
					abs_target = abs(target)
			
					t_data = []
					if(target > 0):
						img = ban_matrix.get_masu(x,y).snippet_img_arr
					else:
						img = komaRecognition.inverse_img_arr(ban_matrix.get_masu(x,y).snippet_img_arr)

					if(img.shape[0] == 0):
						print "aaa"
						print x
						print y

					if(abs_target == koma):
						t_data.append(1)
						print (x,y)
					elif (target != 0):
						t_data.append(0)
					t_data.append(img)
					t_data_list.append(t_data)

					
		else:
			print "waring set masu img"

	max_epoch =15
	for i in xrange(max_epoch):
		err_ave = 0.0
		for l_data in l_data_list:
			(err, learnW) = DNN.batch_learn(learnW, l_data[1], l_data[0])	
			err_ave += err
			#print err
		err_ave /= len(l_data_list)
		print "       err= " + str(err_ave)

		# test
		success = 0
		index = 0
		for t_data in t_data_list:
			ret = DNN.recog(learnW, t_data[1], [t_data[0]])
			if ret["success"] == 1:
				success += 1
			index += 1
		# result
		print str(success) + "/" + str(index)








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



# print をラップする

class LogObj:
	def __init__(self, main_socket = None):

		self.__main_sockets = [main_socket]
		self.default_node_id = 4

	def log(self, line, node_id = 0):
		if node_id == 0:
			node_id = self.default_node_id
		print line
		if(self.__main_sockets[0]):
			self.__main_sockets[0].push_info_lines(line, node_id)

	def warning(self, line, node_id = 0):
		if node_id == 0:
			node_id = self.default_node_id
		locat = self.location(1)
		warn = line + " [:warning in " + str(locat) + " ]" 
		print warn
		if(self.__main_sockets[0]):
			self.__main_sockets[0].push_info_lines(warn, node_id)

	def copy_obj(self, default_id = 4):
		_copy_obj =  LogObj()
		_copy_obj._set_main_socket(self.__main_sockets)
		_copy_obj.default_node_id = default_id
		return _copy_obj

	def main_socket(self):
		return self.__main_sockets[0]

	def set_main_socket(self, main_socket):
		self.__main_sockets[0] = main_socket

	def _set_main_socket(self, main_sockets):
		self.__main_sockets = main_sockets

	def location(self, depth=0):
		frame = inspect.currentframe(depth+1)
		return (frame.f_code.co_filename, frame.f_lineno)

def main():
	global log_obj
	main_socket = None
	apery = None
	wroom = None
	log_obj = LogObj()
	flow = PlayFlow(log_obj)
	flow.start_flow()
	

	while(True):
		# キーボード入力
		key = cv2.waitKey(1) & 0xFF

		panel_node_id = 0
		if(main_socket):
			panel_node_id =  main_socket.pull_panel_node_id() 

		if key == 27 or key == ord('q') or panel_node_id == 4: # esc
			flow.end_flow()
			if(main_socket):
				main_socket.kill_me()
			if(wroom):
				wroom.end()
			break
		elif key == ord("a"):
			if(apery == None):
				log_obj.log("start apery assist", 8)
				apery = apery_call.AperyCall()
				flow.apery_assist(apery)

		elif key == ord("s") or panel_node_id == 8:
			if(apery == None):
				apery = apery_call.AperyCall()
				flow.apery_assist(apery)

			log_obj.log("ask apery", 8)
			ans = flow.get_apery_ans().get_txt(utf8 = True)
			log_obj.log("apery tell you", 8)
			log_obj.log(ans, 8)
		elif key == ord("d") or panel_node_id == 12: # DNN
			log_obj.log("set machine learn on", 12)
			flow.is_learn = True
			if(main_socket):
				main_socket.set_info_status(1, node_id = 12)

		elif key == ord("m"): # main socket
			if(not(main_socket)):
				main_socket = main_sock.main()
				log_obj.set_main_socket(main_socket)
				log_obj.log("main_socket", 4)

		elif key == ord("w") or panel_node_id == 7:
			if(not(wroom)):
				wroom = wroom_master.WroomHost(flow.get_apery_ans, log_obj = log_obj.copy_obj(7))
				
				if(main_socket):
					main_socket.set_info_status(1, node_id = 7)

		elif key == ord("y") or panel_node_id == 11: # wroom 認識(for debug)
			log_obj.log("wroom debug", 11)
			if(apery == None):
				log_obj.log("start apery assist", 8)
				apery = apery_call.AperyCall()
				flow.apery_assist(apery)
			flow.get_apery_ans( by_wroom = True)



	print "successfully ended"

if __name__ == '__main__':
	#main()
	#sengo_test()
	koma_test()





