class Morph:
	surface : str
	base : str
	pos : str
	pos1 : str
	def __init__(self, line: str) -> None:
		word = line.split(' ')
		self.surface = word[0]
		self.base = word[2]
		self.pos = word[3]
		self.pos1 = word[5]
	def print_self(self):
		print(self.surface, self.base, self.pos, self.pos1)
	def check_pos(self, pos:str):
		if pos == self.pos:
			return 1
		return 0
	def check_pos1(self, pos1:str):
		if pos1 == self.pos1:
			return 1
		return 0


class Chunk:
	morphs : list
	dst : int
	srcs : list
	phrase : str
	def __init__(self, morphs, dst, srcs):
		self.morphs = morphs
		self.dst = dst
		self.srcs = srcs
		self.phrase = ''
		for morph in self.morphs:
			self.phrase += morph.surface
	def print_self(self):
		print(self.phrase, end =' ')
	def check_pos(self, pos:str) -> int:
		for morph in self.morphs:
			if morph.pos == pos:
				return 1
		return 0
	def check_pos1(self, pos1:str) -> int:
		for morph in self.morphs:
			if morph.pos1 == pos1:
				return 1
		return 0
	def check_surface(self, surface:str) -> int:
		for morph in self.morphs:
			if morph.surface == surface:
				return 1
		return 0
	def XYphrase(self, XorY:str) -> str:
		XYphrase = ''
		flag = 0
		for morph in self.morphs:
			if morph.pos == '名詞' and flag == 0:
				XYphrase += XorY
				flag += 1
			elif morph.pos == '名詞' and flag >= 1:
				continue
			else:
				XYphrase += morph.surface
		return XYphrase
