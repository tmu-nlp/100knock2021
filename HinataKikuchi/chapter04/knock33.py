from knock30 import analized_dicts as d

A_s_B = []
for i in range(1, len(d) - 1):
	if 'surface' in d[i].keys():
		if d[i]['surface'] == 'の' and d[i - 1]['pos'] == '名詞' and d[i + 1]['pos'] == '名詞':
			A_s_B.append(d[i-1]['surface'] + d[i]['surface'] + d[i+1]['surface'])

print(A_s_B)