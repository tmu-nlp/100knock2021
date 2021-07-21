#!/usr/bin/env bash

<<comment
Japanese-English Subtitle Corpus (JESC)やJParaCrawlなどの翻訳データを活用し，KFTTのテストデータの性能向上を試みよ．
JParaCrawl Chinese-Japanese training set (83k sentences)を利用
comment


fairseq-preprocess -s zh -t ja \
    --trainpref train.sub \
    --validpref valid.sub \
    --destdir data98 \
    --workers 20




fairseq-train data98 \
    --fp16 \
    --save-dir save98_1 \
    --max-epoch 3 \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --clip-norm 1.0 \
    --lr 1e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.1 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 8000 > 98_1.log


fairseq-interactive --path save98_1/checkpoint3.pt data98 < test.sub.ja | grep '^H' | cut -f3 | sed -r 's/(@@ )|(@@ ?$)//g' > 98_1.out