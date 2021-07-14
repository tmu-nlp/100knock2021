# contents are in first_half.ipynb









# import re
# from collections import defaultdict
# import torch 
# def knock80():
#     with open("data/train.txt", "r") as f1:
#         word_dict = {}
#         word_id = 1
#         word_dedict = defaultdict(lambda:0)
#         for line in f1:
#             words = line.strip().split("\t")[0]
#             words = re.sub(",|\.|\"|\'|:|\?|\!","",words).lower().split()
#             for word in words:
#                 word_dedict[word] += 1
#         word_dedict = sorted(word_dedict.items(), key = lambda x:x[1], reverse = True)
#         for key, value in word_dedict:
#             if value != 1:
#                 word_dict[key] = word_id
#                 word_id += 1
#             else:
#                 word_dict[key] = 0
#         with open("word_dict.txt", "w") as f2:
#             for key, value in word_dict.items():
#                 f2.write(f"{key}\t{value}\n")

# def load_word_dict():
#     word_dict = {}
#     with open("word_dict.txt","r") as f3:
#         for line in f3:
#             key, value = line.strip().split("\t")
#             word_dict[key] = value
#     return word_dict

# def convert_w2i(str1):
#     word_dict = load_word_dict()
#     str1 = re.sub(",|\.|\"|\'|:|\?|\!",'',str1).lower()
#     target_word_list = str1.split()
#     word_id_list = []
#     word_list = [key for key, _ in word_dict.items()]
#     for word in target_word_list:
#         if word in word_list:
#             word_id_list.append(int(word_dict[word]))
#         else:
#             word_id_list.append(-1)
#     print(list(word_dict)[:10])
#     return word_id_list
            

# if __name__ == "__main__":
#     knock80()
#     print(convert_w2i("I can be a champion and toya, brosnan jjjjjjjjj to update!"))