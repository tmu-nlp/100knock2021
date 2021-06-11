import tarfile
import os

def spacy_tokenize(src, dst):
    with open(src) as f, open(dst, 'w') as g:
        for x in f:
            x = x.strip()
            x = ' '.join([doc.text for doc in nlp(x)])
            print(x, file=g)

with tarfile.open('en-ja.tar.gz') as tar:
    for f in tar.getmembers():
        if f.name.endswith('txt'):
            text = tar.extractfile(f).read().decode('utf-8')
            break

data = text.splitlines()
data = [x.split('\t') for x in data]
data = [x for x in data if len(x) == 4]
data = [[x[3], x[2]] for x in data]

with open('jparacrawl.ja', 'w') as f, open('jparacrawl.en', 'w') as g:
    for j, e in data:
        print(j, file=f)
        print(e, file=g)

with open('jparacrawl.ja') as f, open('train.jparacrawl.ja', 'w') as g:
    for x in f:
        x = x.strip()
        x = re.sub(r'\s+', ' ', x)
        x = sp.encode_as_pieces(x)
        x = ' '.join(x)
        print(x, file=g)

os.system("subword-nmt apply-bpe -c kyoto_en.codes < jparacrawl.en > train.jparacrawl.en")
os.system(
"fairseq-preprocess -s ja -t en \
    --trainpref train.jparacrawl \
    --validpref dev.sub \
    --destdir data98  \
    --workers 20"
)
os.system(
"fairseq-train data98 \
    --fp16 \
    --save-dir save98_1 \
    --max-epoch 3 \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --clip-norm 1.0 \
    --lr 1e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.1 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 8000 > 98_1.log"
)
os.system("fairseq-interactive --path save98_1/checkpoint3.pt data98 < test.sub.ja | grep '^H' | cut -f3 | sed -r 's/(@@ )|(@@ ?$)//g' > 98_1.out")
spacy_tokenize('98_1.out', '98_1.out.spacy')
os.system("fairseq-score --sys 98_1.out.spacy --ref test.spacy.en")
os.system(
"fairseq-preprocess -s ja -t en \
    --trainpref train.sub \
    --validpref dev.sub \
    --tgtdict data98/dict.en.txt \
    --srcdict data98/dict.ja.txt \
    --destdir data98_2  \
    --workers 20"
)
os.system(
"fairseq-train data98_2 \
    --fp16 \
    --restore-file save98_1/checkpoint3.pt \
    --save-dir save98_2 \
    --max-epoch 10 \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --clip-norm 1.0 \
    --lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 2000 \
    --dropout 0.1 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 8000 > 98_2.log"
)
os.system("fairseq-interactive --path save98_2/checkpoint10.pt data98_2 < test.sub.ja | grep '^H' | cut -f3 | sed -r 's/(@@ )|(@@ ?$)//g' > 98_2.out")
spacy_tokenize('98_2.out', '98_2.out.spacy')
os.system("fairseq-score --sys 98_2.out.spacy --ref test.spacy.en")