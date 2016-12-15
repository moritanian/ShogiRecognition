#!/usr/bin/env python
# -*- coding: utf-8 -*-

import socket
import threading
import time
import datetime
import subprocess
import json
import os
import signal
 
HOST = '127.0.0.1'
PORT = 50007
INTERVAL = 1 # 測定間隔

def kill_me():
    global server, soc
    soc.fin_f = True
    # 自分で相手側のソケットをつくる(自分のソケットを終了するため)
    SocketSlave(back= False)
    server.close()
    print "stop"    


class SocketMaster:

    def __init__(self):
        self.fin_f = False
        self.soc = threading.Timer(0, self.socket_work)
        self.soc.start()

    # サーバを作成して動かす関数
    def socket_work(self):
         
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((HOST, PORT))
        s.listen(1)

     
        while True:
            conn, addr = s.accept()
            if self.fin_f:
                break
            print 'Connected by', addr
            data = conn.recv(1024)
            print data
            obj = json.loads(data)

            if ('get_status' in obj) and  obj['get_status'] == 1:
                ret_obj = [{"node_id" :1, "status": 1}, {"node_id": 2,"status": 0},  {"node_id": 3,"status": 0},]

            elif 'node_id' in obj:
                node_id = obj["node_id"]
                print node_id
                if int(node_id) == 4:
                    print "fin signal"
                    kill_me()
                ret_obj = "node_id " + str(obj["node_id"])
            else:
                c = obj['a'] + obj['b'] + 1
                ret_obj = {"answer": c}
            
            conn.send(json.dumps(ret_obj))
            conn.close()

def SocketSlave(obj= {}, port = 50007, back = True):
    json_str = json.dumps(obj)

    HOST = '127.0.0.1'    # The remote host
    PORT = port # The same port as used by the server
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    s.send(json_str)
    data = "{}"
    if(back):
        data = s.recv(1024)
    s.close()
    #print 'Received', repr(data)
    return json.loads(data)

class Server:
    def __init__(self):
        cmd = "python cgiserver.py"
       # cmd = "python \n import CGIHTTPServer \n CGIHTTPServer.test()"
        self.proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE , stdin = subprocess.PIPE )

        self.log_thread = threading.Timer(0, self.log) 
        self.log_thread.start()
        
    def log(self):
        print "log"
        while True:
            line = self.proc.stdout.readline()
            print line
 
    def close(self):
        self.proc.stdin.write(chr(3)) # ctrl C
        self.proc.stdin.write("\n") # ctrl C
        #line = self.proc.stdout.readline()
        #print line
        os.kill(self.proc.pid,  signal.CTRL_BREAK_EVENT)
        #self.proc.kill()


if __name__ == '__main__':

    # サーバ起動
    server = Server()
     
    # サーバを作成して動かす
    soc = SocketMaster()

    while (True):
        key = raw_input()
        if key == "q":
            kill_me()
            break

    print "fin"



