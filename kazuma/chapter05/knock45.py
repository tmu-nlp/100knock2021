'''
-- knock45 --
今回用いている文章をコーパスと見なし，日本語の述語が取りうる格を調査したい．
動詞を述語，動詞に係っている文節の助詞を格と考え，述語と格をタブ区切り形式で出力せよ．
ただし，出力は以下の仕様を満たすようにせよ．

・動詞を含む文節において，最左の動詞の基本形を述語とする
・述語に係る助詞を格とする
・述語に係る助詞（文節）が複数あるときは，すべての助詞をスペース区切りで辞書順に並べる

「ジョン・マッカーシーはAIに関する最初の会議で人工知能という用語を作り出した。」という例文を考える． 
この文は「作り出す」という１つの動詞を含み，「作り出す」に係る文節は「ジョン・マッカーシーは」，
「会議で」，「用語を」であると解析された場合は，次のような出力になるはずである．

作り出す	で は を

このプログラムの出力をファイルに保存し，以下の事項をUNIXコマンドを用いて確認せよ．

・コーパス中で頻出する述語と格パターンの組み合わせ
・「行う」「なる」「与える」という動詞の格パターン（コーパス中で出現頻度の高い順に並べよ）
'''

from knock41 import get_chunk_sentences
from collections import defaultdict

# # set 使ってやった。unixコマンドの課題の内容的にこれではなさそう。
# sentences = get_chunk_sentences()
# de_dict1 = defaultdict(lambda:set())
# for sentence in sentences:
#     for chunk in sentence:
#         verbs = [a_mor.base for a_mor in sentence[int(chunk.dst)].morphs if a_mor.pos == "動詞"]
#         pps = [a_mor.base for a_mor in chunk.morphs if a_mor.pos == "助詞"]
#         if verbs and pps:
#             de_dict1[verbs[0]] = de_dict1[verbs[0]].union(pps)
# with open("knock45_output.txt", "w") as f:
#     for k,v in de_dict1.items():
#         str1 = " ".join(sorted(v))
#         f.write(f"{k}\t{str1}\n")

sentences = get_chunk_sentences()
with open("knock45_output.txt", "w") as f:
    for sentence in sentences:
        for chunk in sentence:
            verbs = [a_mor.base for a_mor in chunk.morphs if a_mor.pos == "動詞"]
            if not verbs:
                continue
            pps_candidate = [int(a_srcs) for a_srcs in chunk.srcs]
            pps = []
            for i in pps_candidate:
                pps += [a_mor.base for a_mor in sentence[i].morphs if a_mor.pos == "助詞"]
            if pps:
                str1 = " ".join(sorted(pps))
                f.write(f"{verbs[0]}\t{str1}\n")
