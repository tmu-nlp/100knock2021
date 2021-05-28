from knock30 import analized_dicts as d

noun_chain = []

for i in range(len(d)):
	tmp = ''
	while 'pos' in d[i].keys() and d[i]['pos'] == '名詞':
		tmp += d[i]['surface']
		i += 1
	if tmp != '':
		noun_chain.append(tmp)
print(noun_chain)
