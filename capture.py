# -*- coding: utf-8 -*-
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

count = 0

while(True):
    # フレームをキャプチャする
    ret, frame = cap.read()

    # 画面に表示する
    cv2.imshow('frame',frame)

    # キーボード入力待ち
    key = cv2.waitKey(1) & 0xFF

    # qが押された場合は終了する
    if key == ord('q'):
        break
    # sが押された場合は保存する
    if key == ord('s'):
    	count += 1
        path = "init_ban" + str(count) +".jpg"
        cv2.imwrite(path,frame)

# キャプチャの後始末と，ウィンドウをすべて消す
cap.release()
cv2.destroyAllWindows()