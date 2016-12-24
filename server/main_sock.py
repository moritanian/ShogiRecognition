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

class SocketMaster:

    def __init__(self):
        self.fin_f = False
        self.soc = threading.Timer(0, self.socket_work)
        self.soc.start()
        info_lines = []
        for i in range(12):
            info_lines.append({"node_id": i+1, "log": "", "status": 1})
        info_lines[7]["status"] = 0
        info_lines[6]["status"] = 0
        info_lines[10]["status"] = 0
        info_lines[11]["status"] = 0

        self.info_lines = info_lines

        self.__panel_node_id = 0

    def kill_me(self):
        self.fin_f = True
        # 自分で相手側のソケットをつくる(自分のソケットを終了するため)
        SocketSlave(back= False)
        #if(server):
        #    server.close()
        print "stop"    

    # サーバを作成して動かす関数
    def socket_work(self):
         
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((HOST, PORT))
        s.listen(1)

     
        while True:
            conn, addr = s.accept()
            if self.fin_f:
                break
            #print 'Connected by', addr
            self.push_info_lines("Connected by " + str(addr), 9)
            data = conn.recv(1024)
            #print data
            self.push_info_lines(str(data), 9)

            obj = json.loads(data)

            if ('get_status' in obj) and  obj['get_status'] == 1:
                json_str = json.dumps(self.info_lines)
                self.clear_info_lines()
                #ret_obj = [{"node_id" :1, "status": 1}, {"node_id": 2,"status": 0},  {"node_id": 3,"status": 0},]

            elif 'node_id' in obj:
                node_id = obj["node_id"]
                print node_id
                self.__panel_node_id = node_id
                
                #if int(node_id) == 4:
                #    print "fin signal"
                #    self.kill_me()

                #elif int(node_id) == 8: # apery

                ret_obj = "node_id " + str(obj["node_id"])
                json_str = json.dumps(ret_obj)
            else:
                c = obj['a'] + obj['b'] + 1
                ret_obj = {"answer": c}
                json_str = json.dumps(ret_obj)
                
            conn.send(json_str)
            conn.close()

    # control panel に渡すための状態のinfoを蓄える
    def push_info_lines(self, line, node_id = 4):
        self.info_lines[node_id -1]["log"] += line + "\n"

    def set_info_status(self, status, node_id = 4):
        self.info_lines[node_id -1]["status"] = status

    def clear_info_lines(self, node_id = 0):
        index_list = range(len(self.info_lines))
        if node_id != 0:
            index_list = (node_id - 1)
        for i in index_list:
            self.info_lines[i]["log"] = ""

    # panel_node_id 読み出し　よみだされたあとは初期値に戻す 
    def pull_panel_node_id(self):
        ret = self.__panel_node_id
        self.__panel_node_id = 0
        return ret

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

# 2016/12/25 現在　使用していない　サーバは別ウィンドウでたちあげること
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

def main():
    global soc, server
    server = None
    # サーバ起動
    #print "server start"
    #server = Server()
     
    # サーバを作成して動かす
    soc = SocketMaster()

    return soc

if __name__ == '__main__':
    soc = main()
    while (True):
        key = raw_input()
        if key == "q":
            soc.kill_me()
            break

    print "fin"



