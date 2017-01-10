# ShogiRecognition

- detect the shogi board in the real world.
- GUI controll panel in html. You can get the status in the system and manipulate it.
- Original Device which has wifi module ESP-WROOM-02. You can cheat with it.
- Store remote server with shogi board data. EveryWhere you can see the board data with tablet or PC.



## environment development 
- scipy (windows https://sourceforge.net/projects/scipy/?source=typ_redirect)
- numpy
- opencv
- PIL
- requests

# 本番で確認すること
- wroomで接続先のIP 
	UART接続は,青、緑,黄　の順
	
- wroom　host のIp設定
- CGIserver 立ち上げ
- image_e 立ち上げ　コマンド　m, a, w 
- cd Desktop/python/ShogiRecognition
 python image_e.py