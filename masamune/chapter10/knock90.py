'''
# download dataset
tar -zxvf kftt-data-1.0.tar.gz
rm kftt-data-1.0.tar.gz

# preprocess
mkdir data-bin
mkdir data-bin/kftt.ja-en

TEXT=../kftt-data-1.0/data/tok
fairseq-preprocess --source-lang ja --target-lang en \
    --trainpref $TEXT/kyoto-train \
    --validpref $TEXT/kyoto-dev \
    --testpref $TEXT/kyoto-test \
    --destdir data-bin/kftt.ja-en/ \
    --thresholdsrc 5 \
    --thresholdtgt 5 \
    --workers 20
'''