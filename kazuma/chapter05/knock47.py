'''
-- knock47 --
動詞のヲ格にサ変接続名詞が入っている場合のみに着目したい．
46のプログラムを以下の仕様を満たすように改変せよ．

・「サ変接続名詞+を（助詞）」で構成される文節が動詞に係る場合のみを対象とする
・述語は「サ変接続名詞+を+動詞の基本形」とし，文節中に複数の動詞があるときは，最左の動詞を用いる
・述語に係る助詞（文節）が複数あるときは，すべての助詞をスペース区切りで辞書順に並べる
・述語に係る文節が複数ある場合は，すべての項をスペース区切りで並べる（助詞の並び順と揃えよ）

例えば「また、自らの経験を元に学習を行う強化学習という手法もある。」という文から，
以下の出力が得られるはずである．

学習を行う	に を	元に 経験を

'''
import knock40, knock41
# from knock41 import  Chunk
class ExtendChunk(knock41.Chunk):
    def __init__(self, dst, srcs, score):
        super().__init__(dst, srcs)
        self.score = score
from collections import defaultdict

def get_extend_chunk_sentences():
    '''
    sentences = [sentence 1, sentence 2, ...]
    sentence = [chunk 1, cuhnk 2, ...]
    '''
    sentence = []
    sentences = []
    d1 = defaultdict(lambda:[])
    f = knock40.load_file()
    for line in f:
        
        # 文末まできたら、sentenceをsentencesに加え、sentenceとd1を再初期化。
        if line.strip() == "EOS":
            sentences.append(sentence)
            sentence = []
            d1 = defaultdict(lambda:[])

        # 文節情報の行なら、
        elif line.strip("\n").split(" ")[0] == "*":
            line = line.strip("\n").split(" ")
            chunk = ExtendChunk(line[2][:-1], d1[line[1]], float(line[4]))
            d1[line[2][:-1]].append(line[1])
            sentence.append(chunk)

        # 形態素の行なら、
        else :
            chunk.morphs.append(knock40.get_mor(line))

    return sentences

sentences = get_extend_chunk_sentences()
f = open("knock47_output.txt", "w") # インデント減らしたくなったからwith open やめてみた。
for sentence in sentences:
    for chunk in sentence:
        verbs = [a_mor.base for a_mor in chunk.morphs if a_mor.pos == "動詞"]
        if not verbs:
            continue
        pps_candidate = [int(a_srcs) for a_srcs in chunk.srcs]
        pps_chunks_list = []
        list2 = []
        for i in pps_candidate:
            for e_cnt, mor in enumerate(sentence[i].morphs):
                if mor.pos == "助詞" and e_cnt >= 1:
                    if sentence[i].morphs[e_cnt].base == "を" and sentence[i].morphs[e_cnt-1].pos1 == "サ変接続":
                        list2.append([sentence[i].morphs[e_cnt-1].surface, sentence[i].score, "".join([a_mor.surface for a_mor in sentence[i].morphs if a_mor.pos != "記号"])])
            list1 = [a_mor.base for a_mor in sentence[i].morphs if a_mor.pos == "助詞"]
            if not list1:
                continue
            for pp in list1:
                pps_chunks_list += [(pp, "".join([a_mor.surface for a_mor in sentence[i].morphs if a_mor.pos != "記号"]))]
        for e_cnt, j in enumerate(list2):
            max_score = max([k[1] for k in list2])
            if j[1] == max_score:
                verbs[0] = j[0] + "を" + verbs[0]
                pps_chunks_list.remove(("を", j[2]))
        pps_chunks_list.sort()
        if pps_chunks_list and list2:
            str1 = " ".join([i[0] for i in pps_chunks_list])
            str2 = " ".join([i[1] for i in pps_chunks_list])
            f.write(f"{verbs[0]}\t{str1}\t{str2}\n")
f.close()