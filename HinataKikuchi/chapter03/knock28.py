from knock27 import dict_info
import re


pattern1 = r'\{\{'
pattern2 = r'\}\}'
pattern3 = r'\'\''
pattern4 = r'\|'
pattern5 = r':.*'
for key, val in dict_info.items():
	tmp = re.sub(pattern3,'',re.sub(pattern2,'',(re.sub(pattern1, '', val))))
	dict_info[key] = re.sub(pattern5, '', re.sub(pattern4, '', tmp))
