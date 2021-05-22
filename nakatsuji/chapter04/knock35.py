from knock30 import sentences
from collections import defaultdict

count = defaultdict(lambda: 0)
for sen in sentences:
    for word in sen:
        if word['pos'] != '記号':
            count[word['base']] += 1  
count = sorted(count.items(), key=lambda x: x[1], reverse=True)

if __name__ == '__main__':
    for w in count[:5]:
        print(w)