import os

for r in [0.1, 0.3, 0.5]:
    os.system(
    "fairseq-train data95 \
    --fp16 \
    --save-dir save97_1 \
    --max-epoch 10 \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --clip-norm 1.0 \
    --lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 2000 \
    --dropout {} --weight-decay 0.0001 \
    --update-freq 1 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 8000 > 97_{}.log".format(r, str(r)[-1])
    )

for i in [1, 3, 5]:
    os.system("fairseq-score --sys 97_{}.out.spacy --ref test.spacy.en".format(i))