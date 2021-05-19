import re
from knock27 import dic_remove_il
from collections import defaultdict

def remove_mwmu(text):
    pattern = r'\'{2,5}'
    text = re.sub(pattern, '', text)

    pattern = r'\[\[(?:[^|]*?\|)??([^|]*?)\]\]'
    text = re.sub(pattern, r'\1', text)

    pattern = r'https?://[\w!?/\+\-_~=;\.,*&@#$%\(\)\'\[\]]+'
    text = re.sub(pattern, '', text)

    pattern = r'<.+?>'
    text = re.sub(pattern, '', text)

    pattern = r'\{\{(?:lang|仮リンク)(?:[^|]*?\|)*?([^|]*?)\}\}' 
    text = re.sub(pattern, r'\1', text)
    return text

dic_remove_mwmu = defaultdict(str)
for k, v in dic_remove_il.items():
	dic_remove_mwmu[k] = remove_mwmu(v)
dic_remove_mwmu = dict(dic_remove_mwmu)

if __name__ == "__main__":
	for k, v in dic_remove_mwmu.items():
		print(k + ': ' + v)