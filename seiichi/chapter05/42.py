import re
import CaboCha
import pickle
from common import Morph, Chunk

with open("data/all_sent.pkl", "rb") as f:
    all_sent = pickle.load(f)

kakari_list = []
sentence_num = -1
for one_sentence in all_sent:
    sentence_num += 1
    for kakari_moto, kakari_saki in enumerate(one_sentence.dst):
        
        if kakari_saki != "-1": # "-1"は係り先がないことを示す
            moto = all_sent[sentence_num].sentence[int(kakari_moto)]
            saki = all_sent[sentence_num].sentence[int(kakari_saki)]
            
            moto = re.sub(r"\u3000", "", moto)
            saki = re.sub(r"。|、", "", saki)
            
            if  moto != '':
                kakari_list.append(str(moto + "\t" + saki))

print(kakari_list[:10])
