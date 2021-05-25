from knock30 import analized_dicts as d
import collections

nouns = []


for dic in d:
	if 'pos' in dic.keys() and dic['pos'] == '名詞':
		nouns.append(dic['surface'])
sorted_nouns = collections.Counter(nouns).most_common()