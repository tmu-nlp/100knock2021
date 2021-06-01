from knock41 import phrases

def find_another_path(phrase, dst):
	counter = []
	for ch in phrase:
		counter.append(ch.dst)
	if counter.count(dst) > 1:
		return 1
	return 0

def get_noun_path(phrase):
	for ch in phrase:
		if ch.check_pos('名詞') == 1 and int(ch.dst) != -1 :
			if find_another_path(phrase, ch.dst) == 1:
				print(ch.XYphrase('X'), end = ' | ')
			else:
				print(ch.XYphrase('X'), end = ' -> ')
			tmp = phrase[int(ch.dst)]
			if tmp.dst != -1:
				while int(tmp.dst) != -1:
					print(tmp.XYphrase('Y'), end = ' ->')
					tmp = phrase[int(tmp.dst)]
			print(' | ' + tmp.XYphrase('Y'))
# get_noun_path(phrases[1])
for phrase in phrases:
	get_noun_path(phrase)