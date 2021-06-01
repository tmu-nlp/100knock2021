#係り元の文節と係り先の文節のテキストをタブ区切り形式ですべて抽出せよ．ただし，句読点などの記号は出力しないようにせよ．

from knock40 import Morph
from knock41 import Chunks, chunk_list

if __name__ == '__main__':
    f = open('ai.ja.txt.parsed', 'r')
    
    sentences = chunk_list(f)
    phrase_list = []

    for chunk in sentences[2]: #フレーズのリストを作成
        phrase = ''
        for morph in vars(chunk)['morphs']: # morph は一単語分の形態素解析
            if vars(morph)['pos'] != '記号':
                phrase = phrase + vars(morph)['surface']
        phrase_list.append(phrase)

    for i, chunk in enumerate(sentences[2]):
        dst = vars(chunk)['dst']
        if dst != -1:
            print('{}\t{}'.format(phrase_list[i], phrase_list[dst]))

