#!/usr/bin/env bash

<<comment
97. ハイパー・パラメータの調整
ニューラルネットワークのモデルや，そのハイパーパラメータを変更しつつ，開発データにおける
BLEUスコアが最大となるモデルとハイパーパラメータを求めよ．
モデルをtransformerに変更。
comment

DATA=./knock95_bpe
DROPOUTS=(0.1 0.2 0.3)
for N in 'seq 3' ; do

    DROPOUT=$DROPOUTS[$N]
    OUT=./checkpoints/knock97_dropout$DROPOUT
    fairseq-train $DATA \
        --seed 1 \
        --optimizer adam --clip-norm 0.0 --adam-betas '(0.9, 0.98)' \
        --arch transformer \
        --share-decoder-input-output-embed \
        --dropout $DROPOUT \
        --warpup-init-lr 1e-7 \
        --min-lr 1e-9 \
        --update-freq 8 \
        --lr 5e4 --lr-scheduler inverse_aqrt --warmup-updates 2000 \
        --weight-decay 0.0 \
        --max-tokens 4096 \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --max-epoch 10 \
        --save-dir $OUT
done


