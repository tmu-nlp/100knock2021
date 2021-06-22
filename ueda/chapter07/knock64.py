from knock60 import struct_vector

model = struct_vector()
with open(r'C:\Git\questions-words.txt', encoding="utf-8") as f, open(r'C:\Git\questions-words-similar.txt', 'w', encoding="utf-8") as g:
    for line in f:
        line = line.strip().split(" ")
        if len(line) != 4:
            g.write(' '.join(line)+'\n')
            continue
        similar_word = model.most_similar(positive=[line[1], line[2]], negative=[line[0]], topn=1)
        g.write(' '.join(line)+' '+similar_word[0][0]+' '+str(similar_word[0][1])+'\n')
