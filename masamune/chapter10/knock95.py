'''
#前処理
mkdir data-bin/kftt-bpe.ja-en
mkdir ../kftt-data-1.0/data/bpe

BPETEXT = '../kftt-data-1.0/data/bpe'
!fairseq-preprocess --source-lang ja --target-lang en \
    --trainpref $BPETEXT/kyoto-train \
    --validpref $BPETEXT/kyoto-dev \
    --testpref $BPETEXT/kyoto-test \
    --destdir data-bin/kftt-bpe.ja-en/ \
    --thresholdsrc 5 \
    --thresholdtgt 5 \
    --bpe subword_nmt \
    --workers 20

#学習
mkdir checkpoints/kftt-bpe.ja-en
fairseq-train data-bin/kftt-bpe.ja-en \
    --save-dir checkpoints/kftt-bpe.ja-en/ \
    --arch lstm --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --max-epoch 5 \
    --eval-bleu \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric"
 
#翻訳
fairseq-interactive data-bin/kftt-bpe.ja-en \
    --path checkpoints/kftt-bpe.ja-en/checkpoint_best.pt \
    --remove-bpe --bpe subword_nmt --bpe-codes $BPE_CODE\
    < ../kftt-data-1.0/data/tok/kyoto-test.ja \
    | grep '^H' | cut -f3 > 95.out

#評価
!fairseq-score --sys 95.out --ref ../kftt-data-1.0/data/tok/kyoto-test.en
'''