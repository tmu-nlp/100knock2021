#!/usr/bin/env bash

<<comment
93. BLEUスコアの計測
91で学習したニューラル機械翻訳モデルの品質を調べるため，評価データにおけるBLEUスコアを測定せよ．
comment

CUDA_VISIBLE_DEVICE=0 fairseq-generate ./data-bin/kftt.ja-en \
    --path knock91/checkpoint_best.pt \
    --beam 5 --batch-size 128 --remove-bpe | tee /tmp/gen.out

grep ^H /tmp/gen.out | cut -f3- > /tmp/gen.out.sys
grep ^H /tmp/gen.out | cut -f2- > /tmp/gen.out.ref


fairseq-score --sys /tmp/gen.out.sys --ref /tmp/gen.out.ref

