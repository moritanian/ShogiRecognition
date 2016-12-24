#! /usr/bin/env python
# coding:utf-8


import requests


import subprocess


def request_API(text = ",,焼きそばに近いが勝ってもいないし負けてもいない別の物となったのだ　カップ焼きそば・・・このような現象を「カップ焼きそば現象」と名付けた　これは常に身の回りで起こっている o"):
	#text = "スターリングラード攻防戦（スターリングラードこうぼうせん、1942年6月28日 - 1943年2月2日）は、第二次世界大戦の独ソ戦において、ソビエト連邦領内のヴォルガ川西岸に広がる工業都市スターリングラード（現ヴォルゴグラード）を巡り繰り広げられた、ドイツ、ルーマニア、イタリア、ハンガリー、およびクロアチアからなる枢軸軍とソビエト赤軍の戦いである。"

	response = requests.post("https://xtxmqknwfshjevs2:@api.voicetext.jp/v1/tts",
	        {'text': text, "speaker": "show", 'volume': 180})
	#レスポンスオブジェクトのjsonメソッドを使うと、
	    #JSONデータをPythonの辞書オブジェクトに変換して取得できる
	#print (response.content)
	f_name = "temp1.wav"
	f = open(f_name, "wb")
	f.write(response.content)
	f.close() 
	subprocess.call("sox/sox " +  f_name + " -b -u -r 7800 kihu.wav")
	return "kihu.wav"


#request_API()