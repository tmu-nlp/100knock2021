###
from collections import defaultdict
from k40 import Morph, morph2sents
from k41 import Chunk, morph2chunk

if __name__ == "__main__":
    with open("/users/kcnco/github/100knock2021/pan/chapter05/ai.ja1.txt.parsed", "r") as ai:
        ai = ai.readlines()

    ai_morphs = []
    for i in range(len(ai)):
        ai_morphs.append(Morph(ai[i]))

    sent = morph2sents(ai_morphs)[32]
    
    #for sent in sents:
    chunks = morph2chunk(sent)
    for i in range(len(chunks)):
        if "名詞" in chunks[i].show_only_listpos():
            #print(chunks[i].show_only_listpos())
            print(chunks[i].show_only_words(), end="")
            goto = chunks[i].dst
            while goto != None:
                print(" -> "+ chunks[goto].show_only_words(), end="")
                goto = chunks[goto].dst
            print("")
