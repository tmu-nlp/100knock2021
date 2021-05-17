from typing import Pattern
from knock26 import dict_info
import re

pattern1 = r'\[\[.*\]\]'
pattern2 = r'\[\['
pattern3 = r'\]\]'
for key,val in dict_info.items():
	tmp = re.findall(pattern1, val)
	if len(tmp) != 0:
		# print(tmp[0].replace(pattern2,'').replace(pattern3, ''))
		tmp1 =re.sub(pattern2, '', tmp[0])
		dict_info[key]=(re.sub(pattern3, '',tmp1))

