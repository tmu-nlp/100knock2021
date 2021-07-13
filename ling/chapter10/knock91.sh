fairseq-preprocess -s ja -t en \
    --trainpref train.spacy \
    --validpref dev.spacy \
    --destdir data91  \
    --thresholdsrc 5 \
    --thresholdtgt 5 \
    --workers 20
'''
2021-07-06 15:29:52 | INFO | fairseq_cli.preprocess | [ja] Dictionary: 54304 types
2021-07-06 15:30:02 | INFO | fairseq_cli.preprocess | [ja] train.spacy.ja: 440288 sents, 11593158 tokens, 1.16% replaced by <unk>
2021-07-06 15:30:02 | INFO | fairseq_cli.preprocess | [ja] Dictionary: 54304 types
2021-07-06 15:30:05 | INFO | fairseq_cli.preprocess | [ja] dev.spacy.ja: 1166 sents, 26533 tokens, 1.18% replaced by <unk>
2021-07-06 15:30:05 | INFO | fairseq_cli.preprocess | [en] Dictionary: 55592 types
2021-07-06 15:30:14 | INFO | fairseq_cli.preprocess | [en] train.spacy.en: 440288 sents, 12343116 tokens, 1.55% replaced by <unk>
2021-07-06 15:30:14 | INFO | fairseq_cli.preprocess | [en] Dictionary: 55592 types
2021-07-06 15:30:17 | INFO | fairseq_cli.preprocess | [en] dev.spacy.en: 1166 sents, 26287 tokens, 2.6% replaced by <unk>
2021-07-06 15:30:17 | INFO | fairseq_cli.preprocess | Wrote preprocessed data to data91
'''

fairseq-train data91 \
    --fp16 \
    --save-dir save91 \
    --max-epoch 10 \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --clip-norm 1.0 \
    --lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 2000 \
    --update-freq 1 \
    --dropout 0.2 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 8000 > 91.log