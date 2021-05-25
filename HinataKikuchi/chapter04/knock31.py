from knock30 import analized_dicts as d

verb_surface = []
for dic in d:
	if 'pos' in dic.keys() and dic['pos'] == '動詞':
		verb_surface.append(dic['surface'])

# print(verb_surface)