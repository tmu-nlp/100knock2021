from knock30 import parse_mecab
from pprint import pprint


with open('neko.txt.mecab') as f:
    parsed = parse_mecab(f.readlines())
pprint([morph['surface'] for morph in parsed if morph['pos'] == '動詞'])
