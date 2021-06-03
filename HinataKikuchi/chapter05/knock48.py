from knock41 import phrases

def get_noun_path(phrase):
	for ch in phrase:
		if ch.check_pos('名詞') == 1 and int(ch.dst) != -1 :
			print(ch.phrase, end = ' ->')
			tmp = phrase[int(ch.dst)]
			if tmp.dst != -1:
				while int(tmp.dst) != -1:
					print(tmp.phrase, end = ' ->')
					tmp = phrase[int(tmp.dst)]
			print(tmp.phrase)
# get_noun_path(phrases[1])
for phrase in phrases:
	get_noun_path(phrase)

###ANS###
#withを使う必要性について
#with はexitとかで呼ばれてる？
#withあるとちゃんと占めてくれマウス