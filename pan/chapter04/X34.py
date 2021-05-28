#名詞の連接（連続して出現する名詞）を最長一致で抽出せよ．

from X30 import mecab

def get_greedy_n():
    lines = mecab(file)
    t = 0
    longest_N_list = []
    Name_phrase = []
    for line in lines:
        for word in line:
            if word['pos'] == '名詞':
                Name_phrase.append(word['surface'])
            else:
                if len(Name_phrase) > 1:
                    longest_N_list.append("".join(Name_phrase))
                Name_phrase = []
        t += 1
        if t == 10:
            break
    return(longest_N_list)

if __name__ == '__main__':
    file = '/users/kcnco/github/100knock2021/pan/chapter04/neko.txt.mecab'
    print(get_greedy_n())
