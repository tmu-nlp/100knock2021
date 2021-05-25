#動詞の基本形をすべて抽出せよ．

from X30 import mecab

def get_v_base():
    lines = mecab(file)
    t = 0
    V_base = []
    for line in lines:
        for word in line:
            if word['pos'] == '動詞':
                base = word['base']
                V_base.append(base)
        t += 1
        if t == 5:
            break
    return(V_base)

if __name__ == '__main__':
    file = '/users/kcnco/github/100knock2021/pan/chapter04/neko.txt.mecab'
    print(get_v_base())
