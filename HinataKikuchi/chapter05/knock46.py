from knock41 import phrases

def print_case_phrase(phrase):
	for ch in phrase:
		if ch.check_pos('動詞') == 1 and len(ch.srcs) != 0:
			print(ch.phrase, end = '\t')
			for idx in ch.srcs:
				if phrase[idx].check_pos('助詞') == 1:
					for morph in phrase[idx].morphs:
						if morph.check_pos('助詞') == 1:
							print(morph.surface, end = '\t')
			for idx in ch.srcs:
				if phrase[idx].check_pos('助詞') == 1:
					print(phrase[idx].phrase, end = '\t')
			print()

# for phrase in phrases:
# 	print_case_phrase(phrase)
#