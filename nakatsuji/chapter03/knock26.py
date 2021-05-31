#import re
#from knock25 import dic
import re, json
from knock25 import basic_dic
from collections import defaultdict

def remove_mu(text):
	text = re.sub(r'\'{2,5}', '', text)
	return text

dic_remove_mu = defaultdict(str)
for k, v in basic_dic.items():
	dic_remove_mu[k] = remove_mu(v)
dic_remove_mu = dict(dic_remove_mu)

if __name__ == "__main__":
	for k, v in dic_remove_mu.items():
		print(k + ': ' + v)