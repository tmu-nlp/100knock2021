import sys
import re

analized_text_path = './neko.txt.mecab'

analized_dicts = []
with open(analized_text_path, mode='r+') as file:
	for line in file.read().split('\n'):
		analized_dict = {}
		if line != 'EOS' and len(re.findall('.*\t',line)) != 0:
			tmp_list = re.findall('\t.*',line)[0].split(',')
			analized_dict['surface'] =  re.findall('.*\t',line)[0].replace('\t','')
			analized_dict['base'] = tmp_list[-3]
			analized_dict['pos'] = tmp_list[0].replace('\t','')
			analized_dict['pos1'] = tmp_list[1]
		analized_dicts.append(analized_dict)

# print(analized_dicts[5])