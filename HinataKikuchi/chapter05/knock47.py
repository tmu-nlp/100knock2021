from knock41 import phrases

###Q###
#サ変接続名詞がない。サ変名詞はあるけど…

def print_case_phrase(phrase):
	for ch in phrase:
		if ch.check_pos1('サ変名詞') == 1 and len(ch.srcs) != 0 and ch.check_surface('を') and phrase[int(ch.dst)].check_pos('動詞'):
			print(ch.phrase + phrase[int(ch.dst)].phrase, end = '\t')
			for idx in ch.srcs:
				if phrase[idx].check_pos('助詞') == 1:
					for morph in phrase[idx].morphs:
						if morph.check_pos('助詞') == 1:
							print(morph.surface, end = '\t')
			for idx in ch.srcs:
				if phrase[idx].check_pos('助詞') == 1:
					print(phrase[idx].phrase, end = '\t')
			print()
# print_case_phrase(phrases[1])
for phrase in phrases:
	print_case_phrase(phrase)
