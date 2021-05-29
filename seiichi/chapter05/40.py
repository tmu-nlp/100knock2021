import CaboCha
from common import Morph

all_sent = []
sent = []
with open("./data/neko.txt.cabocha") as f:
    for line in f:
        if line[0] == "*":
            next
        if "\t" in line:
            item = line.strip().split("\t")
            try:
                surf = item[0]
                items = item[1].split(",")
            except IndexError:
                next
            if not item == ['記号,空白,*,*,*,*,\u3000,\u3000,']:
                sent.append(Morph(surf, items[6], items[0], items[1]))
        elif "EOS" in line:
            if len(sent):
                all_sent.append(sent)
                sent = []

for item in all_sent[1]:
    print('surface=%s\tbase=%s\tpos=%s\tpos1=%s' % (item.surface, item.base, item.pos, item.pos1) )
