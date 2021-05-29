'''
-- knock48 --
文中のすべての名詞を含む文節に対し，その文節から構文木の根に至るパスを抽出せよ．
 ただし，構文木上のパスは以下の仕様を満たすものとする．

・各文節は（表層形の）形態素列で表現する
・パスの開始文節から終了文節に至るまで，各文節の表現を” -> “で連結する

「ジョン・マッカーシーはAIに関する最初の会議で人工知能という用語を作り出した。」という例文を考える．
 CaboChaを係り受け解析に用いた場合，次のような出力が得られると思われる．

ジョンマッカーシーは -> 作り出した
AIに関する -> 最初の -> 会議で -> 作り出した
最初の -> 会議で -> 作り出した
会議で -> 作り出した
人工知能という -> 用語を -> 作り出した
用語を -> 作り出した

KNPを係り受け解析に用いた場合，次のような出力が得られると思われる．

ジョンマッカーシーは -> 作り出した
ＡＩに -> 関する -> 会議で -> 作り出した
会議で -> 作り出した
人工知能と -> いう -> 用語を -> 作り出した
用語を -> 作り出した

'''

from knock41 import get_chunk_sentences
from collections import defaultdict

def seeker(chunk,sentence,list1,flag):
    if chunk.dst == "-1":
        return list1+["".join([mor.surface for mor in chunk.morphs if mor.pos != "記号"])]

    else:
        if flag:
            if "名詞" not in [mor.pos for mor in sentence[int(chunk.dst)].morphs]:
                return seeker(sentence[int(chunk.dst)], sentence, list1+["".join([mor.surface for mor in chunk.morphs if mor.pos != "記号"])], False)
            return seeker(sentence[int(chunk.dst)], sentence, list1+["".join([mor.surface for mor in chunk.morphs if mor.pos != "記号"])], True)
        else:
            if "名詞" not in [mor.pos for mor in sentence[int(chunk.dst)].morphs]:
                flag = False
                return seeker(sentence[int(chunk.dst)], sentence, list1, False)
            flag = True
            return seeker(sentence[int(chunk.dst)], sentence, list1, True)
        

sentences = get_chunk_sentences()
f = open("knock46_output.txt", "w") # インデント減らしたくなったからwith open やめてみた。
for sentence in sentences:
    for chunk in sentence:
        if "名詞" in [mor.pos for mor in chunk.morphs]:
            list1 = seeker(chunk, sentence, [], True)
            if list1 and len(list1) != 1:
                print(" -> ".join(list1))
