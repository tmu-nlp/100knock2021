from eng_json import ans
import re

#re.MULTILINEを指定しない場合、^は文字列の先頭に、$は文字列の末尾しかマッチしない。
pattern = re.compile(r'''
	^\{\{基礎情報.*?$	# '{{基礎情報'で始まる行
	(.*?)		# キャプチャ対象、任意の0文字以上、非貪欲
	^\}\}$		# '}}'の行
	''', re.MULTILINE + re.VERBOSE + re.DOTALL)


dict_info = {}
pattern1 = r'\|.* ='
pattern2 = r'= .*'
templete_list = pattern.findall(ans['text'])

for content in templete_list[0].split('\n'):
	# print(content)
	field_name = re.findall(pattern1, content)
	field_value = re.findall(pattern2, content)
	if len(field_value) != 0 and len(field_name) != 0:
		dict_info[field_name[0][1:-2]] = field_value[0][2:]

