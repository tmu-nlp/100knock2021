from knock41 import phrases

def print_case(phrase):
	for ch in phrase:
		if ch.check_pos('動詞') == 1 and len(ch.srcs) != 0:
			for idx, ch_morph in enumerate(ch.morphs):
				if idx < len(ch.morphs):
					print(ch_morph.surface, end = '')
				else:
					print(ch_morph.base, end='\t')
			for idx in ch.srcs:
				if phrase[idx].check_pos('助詞') == 1:
					for morph in phrase[idx].morphs:
						if morph.check_pos('助詞') == 1:
							print(morph.surface, end = ' ')
			print()

# print_case(phrases[1])

for phrase in phrases:
	print_case(phrase)