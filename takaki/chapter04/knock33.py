from knock30 import parse_mecab
from pprint import pprint


with open('neko.txt.mecab') as f:
    parsed = parse_mecab(f.readlines())
pprint([parsed[i-1]['surface'] + parsed[i]['surface'] + parsed[i+1]['surface'] for i in range(1, len(parsed)-1) if parsed[i-1]['pos'] == '名詞' and parsed[i]['surface'] == 'の' and parsed[i+1]['pos'] == '名詞'])
