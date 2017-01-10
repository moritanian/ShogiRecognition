#! /usr/bin/env python
# coding:utf-8
# tcp_server

import socket
import threading

#from server import main_sock 

from wroom import send_audio
from wroom import synceAPI


import numpy as np


class WroomHost:

	def __init__(self, get_ans_lambda, log_obj=None):
		self.log_obj = log_obj
		self.init_cmd = "0000"
		self.config_cmd = "0001"
		self.close_cmd = "0010"
		self.ask_cmd = "1000"
		self.kihu_cmd = "1001"
		self.voice_start_cmd = "1010"
		self.voice_go_cmd = "1011"
		self.voice_stop_cmd = "1100"
		self.voice_test_cmd = "1101"

		self.voice_path = "voi4.wav"

		self.connection_num = 0
		self.clients = []

		self.get_ans_lambda = get_ans_lambda

		self.VOICE_MODE = 1
		self.KIHU_MODE = 2
		self.TEST_VOICE_MODE = 3

		self.mode = self.VOICE_MODE
		#self.mode = self.KIHU_MODE
		#self.mode = self.TEST_VOICE_MODE
		
		self.voice_size = 1024

		self.test_hz = 100

		self.__end_f = False
		self.bind_ip = '192.168.179.3' # 下記が使えない場合は指定するしかなさそう
		#self.bind_ip = socket.gethostbyname(socket.gethostname())

		self.bind_port = 9999

		self.server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
		#socket.AF_INETでip4を使うことを指定。socket.SOCK_STREAMでTCPを使うことを指定。

		self.server.bind((self.bind_ip,self.bind_port))
		#自分のIPアドレスとportを設定する。
		#相手のIPアドレスとportを設定する場合は、connectを使うと考えてよい。

		self.server.listen(5)
		#コネクションの最大保存数を設定する。

		if(self.log_obj):
			self.log_obj.log("wroom master start")
			self.log_obj.log('[*]Listening on %s:%d' % (self.bind_ip,self.bind_port))
		else:
			print('[*]Listening on %s:%d' % (self.bind_ip,self.bind_port))

		client_handler = threading.Thread(target=self.server_work)
		 
		client_handler.start()

		self.is_stand_alone = False


		

	def end(self):
		self.__end_f = True
		soc=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		soc.connect((self.bind_ip, self.bind_port))
		soc.send("end")
		soc.close()
		for c in self.clients:
			c.close()
		#self.As.end()

	def server_work(self):
		while True:
			
			client,addr = self.server.accept()	
			if(self.__end_f):
				break

			#bind済みのソケットから、新たなソケットと接続先のアドレスを返す。
			if(self.log_obj):
				self.log_obj.log('[*] Accepted connectoin from: %s:%d' % (addr[0],addr[1]))
			else:
				print('[*]Listening on %s:%d' % (self.bind_ip,self.bind_port))
			client_handler = threading.Thread(target=self.handle_client,args=(client,))
			# threadingを使って、スレッドを生成する。マルチコアに対応させたい場合は,multiprocessingを使えば良い。
			# targetは呼び出す関数(オブジェクト)を指定し、argsはその引数を指定している。
			client_handler.start()
			# 処理を開始する。

	# mode 変更
	def change_mode(self):
		if(self.mode == self.VOICE_MODE):
			self.mode = self.KIHU_MODE
			if(self.log_obj):
				self.log_obj.log("set kihu mode")
		else:
			self.mode = self.VOICE_MODE
			if(self.log_obj):
				self.log_obj.log("set voice mode")

	# TODO クソコードすぎる。処理を分割する
	def handle_client(self, client_socket):
		self.connection_num += 1
		self.clients.append(client_socket)
		if(self.log_obj.main_socket()):
			self.log_obj.main_socket().set_info_status(1, node_id = 11)
		bufsize=1024
		request = client_socket.recv(bufsize)

		if(self.log_obj):
			self.log_obj.log( str('[*] Recived: %s' % request))
		else:
			print(str('[*] Recived: %s' % request))
		#client_socket.send(("Hallo Client!!!\n").encode('utf-8'))

		if(request[0:4] == self.init_cmd):
			client_socket.send(self.config_cmd + "config") # TODO set config params 
			if(self.log_obj):
				self.log_obj.log("send config")
			else:
				print("send config")

			while(not(self.__end_f)):
				try:
					request = client_socket.recv(bufsize)
				except sockt.timeout:
					if(self.log_obj):
						self.log_obj.warning("socket timeout")
						return

				if(self.log_obj):
					self.log_obj.log (str('[*] while : %s' % request))
				else:
					print(str('[*] while : %s' % request))

				if(request[0:4] == self.ask_cmd):
					if(self.log_obj):
						self.log_obj.log("ask cmd")
					else:
						print("ask command")
					
					if(self.mode == self.KIHU_MODE): # bibe info
						if(self.log_obj):
							self.log_obj.log("send kihu")
						else:
							print ("send kihu")

						if(self.is_stand_alone):
							bibe = "1"
						else:	
							kihu = self.get_ans_lambda( by_wroom = True)
							bibe = "0"
							if(kihu.is_pointed_good):
								bibe = "1"

						client_socket.send(self.kihu_cmd + bibe)
						
					
					elif(self.mode == self.VOICE_MODE):  # voice send
						if(self.is_stand_alone):
							text = "..私は、  いち　いち　かく　成る がいいと思います。.."
						else:
							kihu = self.get_ans_lambda()
							text = "..私は、 " + kihu.get_voice_txt() + " がいいと思います。.."
						
						self.voice_path = synceAPI.request_API(text)
						#self.voice_path = "tatta2.wav"
						audioS = send_audio.AudioSender(self.voice_path)
						size = audioS.size
						if(self.log_obj):
							self.log_obj.log("send audio" + str(size))
						client_socket.send(self.voice_start_cmd + str(size))
						while(not(self.__end_f)): # send loop
							voice_data = audioS.callback(self.voice_size)
							if(self.log_obj):
								self.log_obj.log("got voice size = ")
							request = client_socket.recv(bufsize)
							if(self.log_obj):
								self.log_obj.log( str('[*] voice : %s' % request))
							if(request[0:4] == self.voice_go_cmd):
								if(self.log_obj):
									self.log_obj.log("voice go cmd")
								client_socket.send(voice_data)
								x = np.frombuffer(voice_data, dtype= "int8") #numpy.arrayに変換
								#print (x)
								#print (len(x))

							else:		# stop voice data sending
								if(self.log_obj):
									self.log_obj.log("break")
								break

					elif(self.mode == self.TEST_VOICE_MODE):
						client_socket.send(self.voice_test_cmd + str(self.test_hz))
						self.test_hz += 200
						if(self.log_obj):
							self.log_obj.log("test " + str(self.test_hz) + "hz")

					if(False):
						if(self.mode == self.VOICE_MODE):
							self.mode = self.KIHU_MODE
						elif (self.mode == self.KIHU_MODE):
							self.mode = self.TEST_VOICE_MODE
						elif(self.TEST_VOICE_MODE):
							self.mode = self.VOICE_MODE


		client_socket.close()
		self.connection_num -= 1
		self.clients.remove(client_socket)
		print ("closed connection")

if __name__ == '__main__':
	host = WroomHost(None)
	host.is_stand_alone = True # 別モジュールなしで動かす
	while(True):
			s1 = raw_input()      
			if(s1 == "q"):
				if(self.log_obj):
					self.log_obj.log("wroom break")
				self.end()
				break	