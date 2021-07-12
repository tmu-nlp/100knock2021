'''
mkdir checkpoints/kftt-search.ja-en/

#学習
mkdir checkpoints/kftt-search.ja-en/lr_5e-3/
fairseq-train data-bin/kftt.ja-en \
    --save-dir checkpoints/kftt-search.ja-en/lr_5e-3 \
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
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric

#翻訳
mkdir 97
fairseq-interactive data-bin/kftt.ja-en/ \
    --path checkpoints/kftt-search.ja-en/lr_5e-3/checkpoint_best.pt \
    < ../kftt-data-1.0/data/tok/kyoto-test.ja \
    | grep '^H' | cut -f3 > 97/5e-3.out

＃評価 
fairseq-score --sys 97/5e-3.out --ref ../kftt-data-1.0/data/tok/kyoto-test.en

#lr=5e-3
BLEU4 = 16.24, 46.1/21.3/11.7/6.9 (BP=0.967, ratio=0.967, syslen=25862, reflen=26734) #参考

#lr=5e-4
BLEU4 = 12.12, 38.4/15.4/8.0/4.5 (BP=1.000, ratio=1.075, syslen=28745, reflen=26734)

#lr=5e-5 
BLEU4 = 2.27, 21.0/4.1/0.9/0.3 (BP=1.000, ratio=1.239, syslen=33127, reflen=26734) #参考

'''