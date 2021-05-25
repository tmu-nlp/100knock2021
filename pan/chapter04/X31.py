#動詞の表層形をすべて抽出せよ.

from X30 import mecab

def get_v_sur():
    lines = mecab(file)
    t = 0
    V_sur = []
    for line in lines:
        for word in line:
            if word['pos'] == '動詞':
                surface = word['surface']
                V_sur.append(surface)
        t += 1
        if t == 5:
            break
    return(V_sur)

if __name__ == "__main__":
    file = '/users/kcnco/github/100knock2021/pan/chapter04/neko.txt.mecab'
    print(get_v_sur())
