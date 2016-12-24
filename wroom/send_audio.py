#! /usr/bin/env python
# coding:utf-8


import sys
import struct
import os
import threading
from time import sleep
import wave

import numpy as np
#import pyaudio

class AudioSender:

	def __init__(self, path):
		self.wf = wave.open(path, "rb")
		self.size = os.path.getsize(path)
		print ("size = " + str(self.size))
		self.count = 0;

		
	def end(self):
		self.wf.close()
		

	def callback(self, size = 256):
		if(self.count >= self.size):
			return 0
		if(self.count + size >= self.size):
			size = self.size - self.count
		data = self.wf.readframes(size)
		#print(data)

		self.count += size
		#data = np.frombuffer(data, dtype= "int8")
		return data


def sample():
	path = "voi2.wav"
	o = AudioSender(path)

	v = str(o.callback(4))
	x = np.frombuffer(v, dtype= "int8") #numpy.arrayに変換
	#x = np.array(v)
	#print (len(x))
	#print(v[1])
	print (x)

	o.end()

#sample()