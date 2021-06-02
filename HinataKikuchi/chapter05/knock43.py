from knock41 import phrases


symbol = '。、\{\}()（「『」』》〉《〈'

def print_dependence(phrase):
	for ch in phrase:
		if ch.phrase !='' and symbol.find(ch.phrase[0]) == -1 and symbol.find(phrase[int(ch.dst)].phrase[0]) == -1:
			if ch.check_pos('名詞') == 1 and phrase[int(ch.dst)].check_pos('動詞') == 1:
				ch.print_self()
				print('\t', end='')
				phrase[int(ch.dst)].print_self()
				print()

for phrase in phrases:
	print_dependence(phrase)