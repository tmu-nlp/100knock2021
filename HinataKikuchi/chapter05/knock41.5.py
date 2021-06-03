###何か変に感じたので書き直します…###
from knock import Chunk, Morph
import re

def get_src(phrase):
	for idx, ch in enumerate(phrase):
		if ch.dst != -1 and idx not in phrase[int(ch.dst)].srcs:
			phrase[int(ch.dst)].srcs.append(idx)

file_path = './ai.ja.txt.parsed'

Chunk_list = []
phrases = []
symbol = '+*#'
with open(file_path, encoding='utf-8') as f:
	morph_list = []
	dst_list = []
	lines = f.read().split('\n')
	dst = re.search(r'[-+]?\d+', lines[1].split(' ')[1]).group()
	for idx, line in enumerate(lines[2:]):
		if line == 'EOS' and len(morph_list)!=0:
			Chunk_list.append(Chunk(morph_list, dst, []))
			phrases.append(Chunk_list)
			Chunk_list = []
			morph_list = []
			phrases.append(Chunk_list)
		elif line[0] == '*':
			dst = re.search(r'[-+]?\d+', line.split(' ')[1]).group()
			Chunk_list.append(Chunk(morph_list, dst, []))
			morph_list = []
		elif symbol.find(line[0]) == -1 and line != 'EOS':
			morph_list.append(Morph(line))
for phrase in phrases:
	get_src(phrase)
