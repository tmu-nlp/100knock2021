##MeCabを使って形態素に分割する

import MeCab
for src, dst in [
    ('/work/michitaka/100knock_ch10/kftt-data-1.0/data/orig/kyoto-train.ja', '/work/michitaka/100knock_ch10/kftt_for_98/train.tok.ja'),
    ('/work/michitaka/100knock_ch10/kftt-data-1.0/data/orig/kyoto-dev.ja', '/work/michitaka/100knock_ch10/kftt_for_98/dev.tok.ja'),
    ('/work/michitaka/100knock_ch10/kftt-data-1.0/data/orig/kyoto-test.ja', '/work/michitaka/100knock_ch10/kftt_for_98/test.tok.ja'), 
]:
    wakati = MeCab.Tagger('-Owakati')
    with open(src) as f, open(dst, 'w') as out:
        for line in f:
            line = line.strip()
            x = wakati.parse(line).split()
            print(' '.join(x), file=out)