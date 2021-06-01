'''
-- knock49 --
文中のすべての名詞句のペアを結ぶ最短係り受けパスを抽出せよ．
ただし，名詞句ペアの文節番号がiとj（i<j）のとき，係り受けパスは以下の仕様を満たすものとする．

・問題48と同様に，パスは開始文節から終了文節に至るまでの各文節の表現（表層形の形態素列）を” -> “で連結して表現する
・文節iとjに含まれる名詞句はそれぞれ，XとYに置換する

また，係り受けパスの形状は，以下の2通りが考えられる．

・文節iから構文木の根に至る経路上に文節jが存在する場合: 文節iから文節jのパスを表示
・上記以外で，文節iと文節jから構文木の根に至る経路上で共通の文節kで交わる場合: 
　文節iから文節kに至る直前のパスと文節jから文節kに至る直前までのパス，文節kの内容を” | “で連結して表示

「ジョン・マッカーシーはAIに関する最初の会議で人工知能という用語を作り出した。」という例文を考える．
 CaboChaを係り受け解析に用いた場合，次のような出力が得られると思われる．

Xは | Yに関する -> 最初の -> 会議で | 作り出した
Xは | Yの -> 会議で | 作り出した
Xは | Yで | 作り出した
Xは | Yという -> 用語を | 作り出した
Xは | Yを | 作り出した
Xに関する -> Yの
Xに関する -> 最初の -> Yで
Xに関する -> 最初の -> 会議で | Yという -> 用語を | 作り出した
Xに関する -> 最初の -> 会議で | Yを | 作り出した
Xの -> Yで
Xの -> 会議で | Yという -> 用語を | 作り出した
Xの -> 会議で | Yを | 作り出した
Xで | Yという -> 用語を | 作り出した
Xで | Yを | 作り出した
Xという -> Yを

KNPを係り受け解析に用いた場合，次のような出力が得られると思われる．

Xは | Yに -> 関する -> 会議で | 作り出した。
Xは | Yで | 作り出した。
Xは | Yと -> いう -> 用語を | 作り出した。
Xは | Yを | 作り出した。
Xに -> 関する -> Yで
Xに -> 関する -> 会議で | Yと -> いう -> 用語を | 作り出した。
Xに -> 関する -> 会議で | Yを | 作り出した。
Xで | Yと -> いう -> 用語を | 作り出した。
Xで | Yを | 作り出した。
Xと -> いう -> Yを

'''
from knock41 import get_chunk_sentences

def get_path(chunk, sentence, list1):    
    if chunk.dst == "-1":
        return list1
    else:
        if "名詞" not in [mor.pos for mor in sentence[int(chunk.dst)].morphs]:
            return get_path(sentence[int(chunk.dst)], sentence, list1)
        return get_path(sentence[int(chunk.dst)], sentence, list1 + [int(chunk.dst)])

def noun_changer(str1, chunk):
    chunk_word = ""
    for mor in chunk.morphs:
        surface = ""
        if mor.pos == "名詞":
            surface = str1
        elif mor.pos != "記号":
            surface = mor.surface
        chunk_word += surface
    return chunk_word

def get_conflict_num(path1, path2, sentence):
    for i in path1:
        if i in path2:
            return i
    for i, chunk in enumerate(sentence):
        if chunk.dst == "-1":
            return i

def get_print_words(sub, path_list, sentence, chunk_num, path1, path2):
    last_num = get_conflict_num(path1, path2, sentence)
    x = []
    for i in path_list:
        if i == last_num :
            return " -> ".join(x)
        if i == chunk_num:
            x.append(sub)
        else:
            x.append("".join([mor.surface for mor in sentence[i].morphs if mor.pos != "記号"]))
    return " -> ".join(x)


sentences = get_chunk_sentences()
# sentence = sentences[0]
for sentence in sentences:
    for chunk_num_i, chunk_i in enumerate(sentence): # chunk_num_i は現在注目している chunkの番号（番目）
        if "名詞" not in [mor.pos for mor in sentence[int(chunk_num_i)].morphs] : # 名詞以外は無視。
            continue
        i_path_list =  get_path(chunk_i, sentence, [chunk_num_i]) # iの根までの番号のlist。
        for chunk_num_j in range(chunk_num_i+1, len(sentence)): # chunk_num_j はchunk_num_i が比較しているchunkの番号（番目)
            if sentence[int(chunk_num_j)].dst == "-1": # rootは無視。
                continue
            if "名詞" not in [mor.pos for mor in sentence[int(chunk_num_j)].morphs] : # 名詞以外は無視。
                continue
            chunk_j = sentence[chunk_num_j]
            j_path_list = get_path(chunk_j, sentence, [chunk_num_j]) # jの根までの番号のlist。
            surface_i = noun_changer("X", chunk_i)
            surface_j = noun_changer("Y", chunk_j)
            
            # rootまでの経路上に他の名詞句があった場合。
            if chunk_num_j in i_path_list:
                print_list = []
                for i in i_path_list:
                    if i == chunk_num_i:
                        print_list.append(surface_i)
                    elif i == chunk_num_j:
                        print_list.append(surface_j)
                        break
                    else:
                        print_list.append("".join([mor.surface for mor in sentence[i].morphs if mor.pos != "記号"]))
                print(" -> ".join(print_list))

            # rootまでの経路上に他の名詞句がなく、文節kでぶつかる場合。
            else:
                np1 = get_print_words(surface_i, i_path_list, sentence, chunk_num_i, i_path_list, j_path_list)
                np2 = get_print_words(surface_j, j_path_list, sentence, chunk_num_j, i_path_list, j_path_list)
                str1 = "".join([mor.surface for mor in sentence[get_conflict_num(i_path_list, j_path_list, sentence)].morphs if mor.pos != "記号"])
                print(f"{np1} | {np2} | {str1}")
            
# 以下残骸。
# X,Y無視して作ったけど、結構買えないと行けなさそうなのとわかりにくいコードになってる。
# def seeker2(chunk,sentence,list1,end_num):
#     if int(chunk.dst) == end_num:
#         return list1 +[int(chunk.dst)]
#     else:
#         return seeker2(sentence[int(chunk.dst)], sentence, list1+[int(chunk.dst)],end_num)
        
# def get_path_to_end(chunk, sentence, list1):
#     if chunk.dst == "-1":
#         return list1
#     else:
#         return get_path_to_end(sentence[int(chunk.dst)], sentence, list1 + [int(chunk.dst)])

# def trans_num_to_word(sentence, num):
#     return "".join([mor.surface for mor in sentence[num].morphs if mor.pos != "記号"])

# sentences = get_chunk_sentences()
# f = open("knock46_output.txt", "w") # インデント減らしたくなったからwith open やめてみた。
# sentence = sentences[0]
# for e_cnt, chunk in enumerate(sentence):
#     flag = True
#     list1 = get_path_to_end(chunk,sentence,[])
#     last_num = None
#     if list1:
#         last_num = list1[-1]
#     list1 = [e_cnt] + list1[:-1]
#     for chu_num_i in range(e_cnt+1, len(sentence)-1):
#         chunk_1 = sentence[chu_num_i]
#         list2 = get_path_to_end(chunk_1,sentence,[])
#         list2 = [chu_num_i] + list2[:-1]
#         if chu_num_i in list1:
#             list3 = seeker2(chunk,sentence,[],chu_num_i)
#             list3 = [e_cnt] + list3
#             list3 = [trans_num_to_word(sentence,i) for i in list3]
#             print(" -> ".join(list3))``
#         else:
#             np1 = " -> ".join([trans_num_to_word(sentence,i) for i in list1])
#             np2 = " -> ".join([trans_num_to_word(sentence,i)for i in list2])
#             str1 = "".join([mor.surface for mor in sentence[last_num].morphs])
#             print(f"{np1} | {np2} | {str1}")
