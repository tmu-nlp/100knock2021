'''
45のプログラムを改変し，述語と格パターンに続けて項（述語に係っている
文節そのもの）をタブ区切り形式で出力せよ．45の仕様に加えて，
以下の仕様を満たすようにせよ．

・項は述語に係っている文節の単語列とする（末尾の助詞を取り除く必要はない）
・述語に係る文節が複数あるときは，助詞と同一の基準・順序でスペース区切りで並べる

「ジョン・マッカーシーはAIに関する最初の会議で人工知能
という用語を作り出した。」という例文を考える． 
この文は「作り出す」という１つの動詞を含み，
「作り出す」に係る文節は「ジョン・マッカーシーは」，「会議で」，「用語を」
であると解析された場合は，次のような出力になるはずである．

作り出す	で は を	会議で ジョンマッカーシーは 用語を

'''

from knock41 import get_chunk_sentences
from collections import defaultdict

sentences = get_chunk_sentences()
f = open("knock46_output.txt", "w") # インデント減らしたくなったからwith open やめてみた。
for sentence in sentences:
    for chunk in sentence:
        verbs = [a_mor.base for a_mor in chunk.morphs if a_mor.pos == "動詞"]
        if not verbs:
            continue
        pps_candidate = [int(a_srcs) for a_srcs in chunk.srcs]
        pps_chunks_list = []
        for i in pps_candidate:
            list1 = [a_mor.base for a_mor in sentence[i].morphs if a_mor.pos == "助詞"]
            if not list1:
                continue
            for pp in list1:
                pps_chunks_list += [(pp, "".join([a_mor.surface for a_mor in sentence[i].morphs if a_mor.pos != "記号"]))]
        pps_chunks_list.sort()
        if pps_chunks_list:
            str1 = " ".join([i[0] for i in pps_chunks_list])
            str2 = " ".join([i[1] for i in pps_chunks_list])
            f.write(f"{verbs[0]}\t{str1}\t{str2}\n")
f.close()