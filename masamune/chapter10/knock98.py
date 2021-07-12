import os

with open('./jesc/split/train') as train_f:
    with open('./jesc/split/train.ja', 'w') as ja_f, open('./jesc/split/train.en', 'w') as en_f:
        for line in train_f:
            en, ja = line.split('\t')
            ja_f.write(ja)
            en_f.write(en+'\n')

os.system("mkdir data-bin/jesc.ja-en")
os.system(
"fairseq-preprocess --source-lang ja --target-lang en \
    --trainpref jesc/split/train \
    --validpref ../kftt-data-1.0/data/tok/kyoto-dev \
    --testpref ../kftt-data-1.0/data/tok/kyoto-test \
    --destdir data-bin/jesc.ja-en/ \
    --bpe subword_nmt \
    --thresholdsrc 5 \
    --thresholdtgt 5 \
    --workers 20"
)
 
os.system("mkdir checkpoints/jesc.ja-en/")
os.system(
"fairseq-train data-bin/jesc.ja-en \
    --save-dir checkpoints/jesc.ja-en/ \
    --arch lstm --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --max-epoch 5 \
    --eval-bleu \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric"
)
