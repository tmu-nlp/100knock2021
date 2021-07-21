#!/usr/bin/env bash


<<comment
90. データの準備
機械翻訳のデータセットをダウンロードせよ．訓練データ，開発データ，評価データを整形し，必要に応じてトークン化などの前処理を行うこと．
ただし，この段階ではトークンの単位として形態素（日本語）および単語（英語）を採用せよ．
参考 https://github.com/pytorch/fairseq/tree/master/examples/translation
comment

# KFTTファイルをダウンロードして解凍

# wget http://www.phontron.com/kftt/download/kftt-data-1.0.tar.gz
tar xzvf ./data/kftt-data-1.0.tar.gz

# fairseq-preprocessで前処理：binarize the data
# src suffix,tgt suffix,data prefix, data storage location
TEXT=./kftt-data-1.0/data/tok
fairseq-preprocess --source-lang ja --target-lang en \
    --trainpref $TEXT/kyoto-train \
    --valiadpref $TEXT/kyoto-dev \
    --testpref $TEXT/kyoto-test \
    --destdir ./data/bin/91_sub \
    --thresholdsrc 5 \ --thresholdtgt 5 \
    --workers 20
