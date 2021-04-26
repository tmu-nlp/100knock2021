word1 = "paraparaparadise"
word2 = "paragraph"

word1_bigram_set = set()
word2_bigram_set = set()
n = 2

for i in range(0, len(word1)):
    word1_bigram_set.add(word1[i:i+n])
for i in range(0, len(word2)):
    word2_bigram_set.add(word2[i:i+n])


print("X：", word1_bigram_set)
print("Y：", word2_bigram_set)
print("和集合：", word1_bigram_set.union(word2_bigram_set))
print("積集合：", word1_bigram_set.intersection(word2_bigram_set))
print("差集合：", word1_bigram_set.difference(word2_bigram_set))