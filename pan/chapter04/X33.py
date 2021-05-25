#2つの名詞が「の」で連結されている名詞句を抽出せよ．

from X30 import mecab

def get_n_phrase():
    lines = mecab(file)
    t = 0
    N_phrase = []
    for line in lines:
        for i in range(1, len(line) - 1):
            if line[i]['surface'] == 'の' \
                    and line[i - 1]['pos'] == '名詞' \
                    and line[i + 1]['pos'] == '名詞':
                phrase = line[i-1]['surface'] + line[i]['surface'] + line[i+1]['surface']
                N_phrase.append(phrase)
        t += 1
        if t == 10:
            break
    return(N_phrase)

if __name__ == '__main__':
    file = '/users/kcnco/github/100knock2021/pan/chapter04/neko.txt.mecab'
    print(get_n_phrase())
