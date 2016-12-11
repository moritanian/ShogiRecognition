#!/usr/bin/env python
# -*- coding: utf-8 -*-

import urllib
import json

def Get(url, params = {}):
	url += "?{0}".format( urllib.urlencode( params))
	localhost = urllib.urlopen(url)
	return  json.loads(localhost.read())


def Post(url, params = {}):
	print url
	params = urllib.urlencode(params)
	print params
	f = urllib.urlopen(url, params)
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
