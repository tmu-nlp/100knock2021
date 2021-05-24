from knock30 import parse_mecab
from pprint import pprint
from collections import defaultdict


def word_count(lines):
    parsed = parse_mecab(lines)
    words  = defaultdict(int)
    for morph in parsed:
        words[morph['base']] += 1
    return sorted(words.items(), key=lambda x:x[1], reverse=True)

if __name__ == '__main__':
    with open('neko.txt.mecab') as f:
        pprint(word_count(f.readlines()))
