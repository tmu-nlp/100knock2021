import CaboCha
import pickle
from common import Morph, Chunk

all_sent = []
one_sent = []
one_chank = []
one_morph = []
count = 0
num_list = []
dst_list = []
surf_list = []
srcs_list = []
sen_chunk_morph = []
sentence_list = []
sentence_list_temp = []
temp = []
morph_list = []


with open('data/neko.txt.cabocha', encoding='utf-8') as f:
    for line in f:
        
        if line[0] == "*":
            temp = []
            if len(sentence_list_temp) > 0:
                for item in sentence_list_temp:
                    temp.append(item)
                sentence_list.append("".join(temp))
                morph_list.append(one_morph)
                one_morph = []
                sentence_list_temp = []
                temp = []
            temp1 = line[:-1].split(" ")
            num_list.append(temp1[1])
            dst_list.append(temp1[2][:-1])
        
        elif "\t" in line:
            item = line.strip().split("\t")
            try:
                surf = item[0]
                items = item[1].split(",")
            except IndexError:
                next
            if item == ['記号,空白,*,*,*,*,\u3000,\u3000,']:
                surf = "\u3000"
            one_morph.append(Morph(surf, items[6], items[0], items[1]))
            sentence_list_temp.append(surf)
                
        elif "EOS" in line:
            temp = []
            if len(sentence_list_temp) > 0: 
                for item in sentence_list_temp:
                    temp.append(item)
                sentence_list.append("".join(temp))
                morph_list.append(one_morph)
                one_morph = []
                sentence_list_temp = [] 
                temp = []
            if len(morph_list) == 0:
                one_sent = [] ; dst_list = [] ; one_morph = [] ; one_chank = [] ; srcs_list = [] ; sentence_list = [] ; sentence_list_temp = [] # リセット
                morph_list = []
                next
            chunk_len = len(dst_list)
            for i in range(chunk_len):
                if str(i) in dst_list: 
                    srcs_list.append([str(s)  for s, x in enumerate(dst_list) if x == str(i)])
                else:
                    srcs_list.append(["-1"])
                
            sentence = "".join(sentence_list)
            if len(sentence) > 0:
                all_sent.append(Chunk(sentence_list, morph_list, dst_list, srcs_list))
            
            one_sent = [] ; dst_list = [] ; one_morph = [] ; one_chank = [] ; srcs_list = [] ; sentence_list = [] ; sentence_list_temp = []
            morph_list = []

with open("data/all_sent.pkl", "wb") as f:
    pickle.dump(all_sent, f)            
