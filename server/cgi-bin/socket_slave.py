#!/usr/bin/env python
# -*- coding: utf-8 -*-

import socket
import json
 

def SocketSlave(obj= {}, port = 50007):
	json_str = json.dumps(obj)

	HOST = '127.0.0.1'    # The remote host
	PORT = port # The same port as used by the server
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.connect((HOST, PORT))
	s.send(json_str)
	data = s.recv(1024)
	s.close()
	#print 'Received', repr(data)
	#return json.loads(data)
	return data # json string