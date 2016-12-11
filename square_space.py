#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class Masu:

	def __init__(self, x, y, s_img = [], _koma = 0, s_img_pos = [0,0]):
		self.set(x, y, s_img, _koma, s_img_pos)

	def set(self,  x, y, s_img = [], _koma = 0, s_img_pos = [0,0]):
		if(s_img == []):
			s_img = np.array([])
		self.x_pos = x
		self.y_pos = y
		self.snippet_img = s_img
		self.koma = _koma
		self.is_koma = not(_koma == 0) # 駒があるか
		self.snippet_img_pos = s_img_pos
		self.snippet_img_size = np.array(self.snippet_img).shape # snippet の元サイズ

		self.snippet_arr_size = 0
		self.snippet_img_arr = None

	def kihu_pos(self):
		return [9 - y, x + 1]
