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
 
a = int(query['a'][0]) #データaを整数として読み込む
b = int(query['b'][0]) #データbを整数として読み込む

obj = {"a":a, "b":b}
ret = socket_slave.SocketSlave(obj)

print "Content-Type:text/javascript"
print
print "callback(" + ret + ");"