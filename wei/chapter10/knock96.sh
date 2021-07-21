#!/usr/bin/env bash
<<comment
96. 学習過程の可視化
# Tensorboardなどのツールを用い，ニューラル機械翻訳モデルが学習されていく過程を可視化せよ．
可視化する項目としては，学習データにおける損失関数の値とBLEUスコア，
開発データにおける損失関数の値とBLEUスコアなどを採用せよ.
comment
DATA=data-bin/kftt.ja-en

# train the model with subwords
CUDA_VISIBLE_DEVICE=0 fairseq-train $DATA \
    --fp16 \
    --save-dir ./knock95_bpe \
    --bpe subword_nmt \
    --max-epoch 10 \
    --optimizer adam --clip-norm 1.0 --adam-batas'(0.9, 0.98)'\
    --arch lstm \
    --share-decoder-input-output-embed \
    --dropout 0.1 --weight-decay 0.0001 \
    --lr 1e-5 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --weight-decay 0.0001 \
    --max-tokens 4096 \
    --update-freq 1 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --eval-bleu \
    --eval-blue-args'{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --tensorboard-logdir ./knock96.log


