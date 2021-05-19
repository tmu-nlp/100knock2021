import re
from knock26 import dic_remove_mu
from collections import defaultdict

def remove_il(text):
    pattern = r'\[\[(?:[^:\]]+?\|)?([^:]+?)\]\]'
    text = re.sub(pattern, r'\1', text)
    return text

dic_remove_il = defaultdict(str)
for k, v in dic_remove_mu.items():
	dic_remove_il[k] = remove_il(v)
dic_remove_il = dict(dic_remove_il)

if __name__ == "__main__":
	for k, v in dic_remove_il.items():
		print(k + ': ' + v)