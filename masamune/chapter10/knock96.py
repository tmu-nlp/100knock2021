'''
mkdir checkpoints/kftt-bpe-tensorboard.ja-en
fairseq-train data-bin/kftt-bpe.ja-en \
    --save-dir checkpoints/kftt-bpe-tensorboard.ja-en/ \
    --tensorboard-logdir checkpoints/kftt-bpe-tensorboard.ja-en/ \
    --arch lstm --share-decoder-input-output-embed \
    --bpe subword_nmt \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --max-epoch 3 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric

tensorboard --logdir ./checkpoints/kftt-bpe-tensorboard.ja-en/
''' 