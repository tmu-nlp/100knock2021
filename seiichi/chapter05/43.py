import re
import CaboCha
import pickle
from common import Morph, Chunk

with open("data/all_sent.pkl", "rb") as f:
    all_sent = pickle.load(f)

kakari_list = []

for sentence_num, one_sentence in enumerate(all_sent):
    for kakari_moto, kakari_saki in enumerate(one_sentence.dst):
        if kakari_saki != "-1":
            flag_moto = False
            for i in range(len(all_sent[sentence_num].morphs[int(kakari_moto)])):
                if all_sent[sentence_num].morphs[int(kakari_moto)][i].pos == "名詞":
                    flag_moto = True
                    break
            flag_saki = False
            for i in range(len(all_sent[sentence_num].morphs[int(kakari_saki)])):
                if all_sent[sentence_num].morphs[int(kakari_saki)][i].pos == "動詞":
                    flag_saki = True
                    break
            if flag_moto == True and flag_saki == True:            
                moto = all_sent[sentence_num].sentence[int(kakari_moto)] 
                saki = all_sent[sentence_num].sentence[int(kakari_saki)]            
                moto = re.sub(r"\u3000", "", moto)
                saki = re.sub(r"。|、", "", saki)
                if  moto != '':
                    kakari_list.append(str(moto + "\t" + saki))

print(kakari_list[10:20])                    
