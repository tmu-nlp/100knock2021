###
from collections import defaultdict
from k40 import Morph, morph2sents
from k41 import Chunk, morph2chunk

#for each sentence
def verbframe2(chunks, verbframedict):
    noun = set()
    verb = set()
    for i in range(len(chunks)):
        for morph in chunks[i].morphs:
            noun.add(morph.pos)
            if "名詞" in noun:
                goto = chunks[i].dst
                if goto != None:
                    for morph2 in chunks[goto].morphs:
                        verb.add(morph2.pos)
        if "動詞" in verb:
            adp = chunks[i].show_base_for_X("助詞")
            if adp != None:
                verbbase = chunks[goto].show_base_for_X("動詞")
                verbframedict[verbbase].append((adp, chunks[i].show_only_words()))
        noun = set()
        verb = set()
    return verbframedict


if __name__ == "__main__":
    with open("/users/kcnco/github/100knock2021/pan/chapter05/ai.ja1.txt.parsed", "r") as ai:
        ai = ai.readlines()

    ai_morphs = []
    for i in range(len(ai)):
        ai_morphs.append(Morph(ai[i]))

    sents = morph2sents(ai_morphs)
    frames = defaultdict(list)
    for sent in sents:
        dep = morph2chunk(sent)
        verbframe2(dep, frames)

    #コーパス中で頻出する述語と格パターンの組み合わせ
    for key, value in sorted(frames.items(), key=lambda x:len(x[1]), reverse=True)[:10]:
        output = defaultdict(list)
        for elem in value:
            output[elem[0]].append(elem[1])
        print(key, end="\t")
        print(" ".join(output.keys()), end="\t")
        print(" ".join([" ".join(elem) for elem in output.values()]))
