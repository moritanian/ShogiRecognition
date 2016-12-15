#!/usr/bin/env python
# -*- coding: utf-8 -*-

# js <=> python の通信例
#  python cgiserver.py # サーバを立てる
#  python .\sock1.py    # python側ソケット
# http://localhost:8000/jsonp.html にアクセス


#GETメソッドでデータを受けとるおまじない
import os
import cgi
import subprocess


import socket_slave

#cmd = "python sock2.py"
#subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE , stdin = subprocess.PIPE )


if 'QUERY_STRING' in os.environ:
    query = cgi.parse_qs(os.environ['QUERY_STRING'])
else:
    query = {}
#おまじない ここまで

get_status = 0
if('get_status' in query):
	get_status = int(query['get_status'][0]) 

node_id = 0
if("node_id" in query):
	node_id = int(query['node_id'][0])

obj = {"get_status":get_status, "node_id":node_id}
ret = socket_slave.SocketSlave(obj)


print "Content-Type:text/javascript"
print
if get_status == 0:
	print "callback("+ ret +");"
else:
	print "get_status_callback("+ ret +");"