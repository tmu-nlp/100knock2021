import re
target_sentence = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
word_list = re.split("[ ,.]",target_sentence)

# word_num_list = []
# for word in word_list:
#     word_num_list.append(len(word))
word_num_list = [len(word) for word in word_list]

word_num_list = [i for i in word_num_list if i != 0]
print(word_num_list)