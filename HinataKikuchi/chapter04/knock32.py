from knock30 import analized_dicts as d

verb_base = []
for dic in d:
	if 'pos' in dic.keys() and dic['pos'] == '動詞':
		verb_base.append(dic['base'])

# print(verb_base)

###ANS###
#