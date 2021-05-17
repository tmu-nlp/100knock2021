from knock25 import dict_info

for val in dict_info.values():
	if val.find('\'\'') != -1:
		val = val.replace('\'',' ')

