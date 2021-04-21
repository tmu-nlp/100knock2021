import re
top1_indexes = [1,5,6,7,8,9,15,16,19]
target_sentence =  "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
word_list = re.split("[ ,.]",target_sentence)
word_list = [word for word in word_list if word != ""]
answer_dict = {}
for i in range(1,len(word_list) + 1):
    if i in top1_indexes:
        answer_dict[i] = word_list[i-1][0]
    else:
        answer_dict[i] = word_list[i-1][:2]
print(answer_dict)