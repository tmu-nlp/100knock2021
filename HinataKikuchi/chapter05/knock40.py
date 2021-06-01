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


file_path = './ai.ja.txt.parsed'

phrases = []
morph_list = []
symbol = '*+E#'
with open(file_path) as f:
	for idx, line in enumerate(f.read().split('\n')):
		if idx < 10:
			print(line)
		if symbol.find(line[0]) == -1:
			morph_list.append(Morph(line))
		if line == 'EOS' and len(morph_list) != 0:
			phrases.append(morph_list)
			morph_list = []

for phrase in phrases[0]:
	print(phrase.surface, end='')
