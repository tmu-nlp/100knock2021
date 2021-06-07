from knock40 import Morph
from knock41 import Chunks, chunk_list

if __name__ == '__main__':
    f = open('ai.ja.txt.parsed', 'r')
    sentences = chunk_list(f)

    n = 0

    for chank in sentences[2]:
        #iのパス
        if chank.have('名詞') == True:
            i = chank.connect()
            x = chank.connect_change_noun_to('X')
            next_chank = sentences[2][chank.dst]
            while True:
                if next_chank.dst == -1:
                    i = i + ' -> ' + next_chank.connect()
                    x = x + ' -> ' + next_chank.connect()
                    n += 1
                    break
                i = i + ' -> ' + next_chank.connect()
                x = x + ' -> ' + next_chank.connect()
                next_chank = sentences[2][next_chank.dst]
            
            #jのパスを全探索
            for chankJ in sentences[2][n:]:
                if chankJ.have('名詞') == True:
                    j = chankJ.connect()
                    y = chankJ.connect_change_noun_to('Y')
                    next_chankJ = sentences[2][chankJ.dst]
                    while True:
                        if next_chankJ.dst == -1:
                            j = j + ' -> ' + next_chankJ.connect()
                            y = y + ' -> ' + next_chankJ.connect()
                            break
                        j = j + ' -> ' + next_chankJ.connect()
                        y = y + ' -> ' + next_chankJ.connect()
                        next_chankJ = sentences[2][next_chankJ.dst]

                #出力
                if i != j:
                    if j in i: #場合1
                        ans = x.replace(j, '') + y.split(' -> ')[0] #iからjへのパス
                        print(ans)
                    else:
                        for m in range(1, len(j)): #jの部分集合を探索
                            if j[m:] in i: #場合2(jの部分集合がiに含まれるかどうか)
                                ans = x.replace(j[m:], '') + ' | ' + y.replace(j[m:], '') + ' | ' + j[m:].split(' -> ')[1]
                                print(ans)
                                break