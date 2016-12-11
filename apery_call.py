#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess

class AperyCall:
	def __init__(self):
		cmd = "D:/ukamuse_sdt4/bin/ukamuse_sdt4_bmi2.exe"
		self.proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE , stdin = subprocess.PIPE )
		self.send_cmd("usi", "usiok")
		self.send_cmd("isready")
		self.send_cmd("usinewgame")
		self.move_list = []	
	

	def move_board(self, move):
		self.move_list.append(move)

	# 待った！
	def wait_board(self, time = 1):
		for i in range(time):
			self.move_list.pop()

	def _send_board(self):
		cmd = "position startpos moves "
		for move in self.move_list:
			cmd += " " + move
		self.send_cmd(cmd)

	def get_answer(self):
		self._send_board()
		print "fin send"
		ret = self.send_cmd("go", "bestmove")
		split = ret[-1].split(" ")
		return [split[1], split[3].strip()] # strip は改行コード削除用


	def send_cmd(self, cmd, wait_until = ""):
		print cmd
		self.proc.stdin.write(cmd + "\n")
		lines = []
		line = ""
		while True:
			if(wait_until == "" or wait_until in line):
				break
			line = self.proc.stdout.readline()
			print line
			lines.append(line)
		
		return lines

	def end_proc(self):
		self.proc.terminate()
		print "apery fininished" 

def sample():
	apery_call = AperyCall()
	apery_call.move_board("7g7f")
	print apery_call.get_answer()

	print "fin"

#sample()