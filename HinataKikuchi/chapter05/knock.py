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

