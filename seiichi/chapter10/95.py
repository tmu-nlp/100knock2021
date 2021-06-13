import os
import re
import spacy
import sentencepiece as spm
import matplotlib.pyplot as plt

# ja
spm.SentencePieceTrainer.Train('--input=./data/kftt-data-1.0/data/orig/kyoto-train.ja --model_prefix=kyoto_ja --vocab_size=16000 --character_coverage=1.0')
sp = spm.SentencePieceProcessor()
sp.Load('kyoto_ja.model')

for src, dst in [
    ('kftt-data-1.0/data/orig/kyoto-train.ja', 'train.sub.ja'),
    ('kftt-data-1.0/data/orig/kyoto-dev.ja', 'dev.sub.ja'),
    ('kftt-data-1.0/data/orig/kyoto-test.ja', 'test.sub.ja'),
]:
    with open(src) as f, open(dst, 'w') as g:
        for x in f:
            x = x.strip()
            x = re.sub(r'\s+', ' ', x)
            x = sp.encode_as_pieces(x)
            x = ' '.join(x)
            print(x, file=g)

# en
os.system("subword-nmt learn-bpe -s 16000 < kftt-data-1.0/data/orig/kyoto-train.en > kyoto_en.codes")
for tar in ["train", "dev", "test"]:
    os.system(f"subword-nmt apply-bpe -c kyoto_en.codes < kftt-data-1.0/data/orig/kyoto-{tar}.en > {tar}.sub.en")

os.system(
"fairseq-preprocess -s ja -t en \
    --trainpref train.sub \
    --validpref dev.sub \
    --destdir data95  \
    --workers 20"
)

os.system(
"fairseq-train data95 \
    --fp16 \
    --save-dir save95 \
    --max-epoch 10 \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --clip-norm 1.0 \
    --lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 2000 \
    --update-freq 1 \
    --dropout 0.2 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 8000 > 95.log"
)

os.system("fairseq-interactive --path save95/checkpoint10.pt data95 < test.sub.ja | grep '^H' | cut -f3 | sed -r 's/(@@ )|(@@ ?$)//g' > 95.out")

def spacy_tokenize(src, dst):
    with open(src) as f, open(dst, 'w') as g:
        for x in f:
            x = x.strip()
            x = ' '.join([doc.text for doc in nlp(x)])
            print(x, file=g)

spacy_tokenize('95.out', '95.out.spacy')

os.system("fairseq-score --sys 95.out.spacy --ref test.spacy.en")

os.system("bash 95_1.sh")

for i in range(1, 11):
    spacy_tokenize(f'95.{i}.out', f'95.{i}.out.spacy')

os.system("bash 95_2.sh")

xs = range(1, 11)
ys = [read_score(f'95.{x}.score') for x in xs]
plt.plot(xs, ys)
plt.show()