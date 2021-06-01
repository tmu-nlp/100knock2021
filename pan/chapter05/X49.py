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
    #rel = []
    chunks = morph2chunk(sent)
    for i in range(len(chunks)):
        for j in range(1+i, len(chunks)):
            if "名詞" in chunks[i].show_only_listpos() and "名詞" in chunks[j].show_only_listpos():
                print(chunks[i].replace_for_X("X"), end="")
                goto = chunks[i].dst
                while goto not in {None,i} and "名詞" in chunks[goto].show_only_listpos():
                    if goto == j:
                        print(" -> " + chunks[goto].replace_for_X("Y"))
                        break
                    else:
                        print(" -> " + chunks[goto].show_only_words(), end="")
                    goto = chunks[goto].dst
                if goto!=j:
                    print(" | "+ chunks[j].replace_for_X("Y"), end="")
                    goto = chunks[j].dst
                    while goto != None and "名詞" in chunks[goto].show_only_listpos():
                        print(" -> " + chunks[goto].show_only_words(), end="")
                        goto = chunks[goto].dst
                    print(" | "+chunks[goto].show_only_words())
