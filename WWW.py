#!/usr/bin/env python
# -*- coding: utf-8 -*-

import urllib
import json

def Get(url, params = {},  timeout=5):
	url += "?{0}".format( urllib.urlencode( params))
	try:
		localhost = urllib.urlopen(url)
	except  IOError:
		print ("Get URL err!!")
		return None
	return  json.loads(localhost.read())


def Post(url, params = {},  timeout=5):
	print url
	params = urllib.urlencode(params)
	print params
	try:
		f = urllib.urlopen(url, params)
	except  IOError :
		print ("POST URL err")
		return None
	json_str = f.read()
	try:
   		ret = json.loads(json_str)
	except ValueError:
		print "json parse Err"
   		print json_str
   		exit()

   	return ret


# localhost = urllib.urlopen("http://localhost/flyby/api/test.json")
#getパラメータ
#param = [
#    ( "id", 0),
#    ( "param", "dammy"),
#]
