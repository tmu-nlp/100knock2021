'''
fairseq-preprocess -s ja -t en \
    --trainpref kftt_for_98/train.tok \
    --validpref kftt_for_98/dev.tok \
    --destdir data98_kftt_baseline  \
    --thresholdsrc 5 \
    --thresholdtgt 5 \
    --workers 20
'''