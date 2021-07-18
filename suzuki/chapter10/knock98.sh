fairseq-preprocess --source-lang ja --target-lang en \
    --trainpref jesc_pre/train --validpref jesc_pre/dev --testpref jesc_pre/test \
    --destdir /clwork/daisuke/data/jesc_data \
    --workers 20

CUDA_VISIBLE_DEVICES=0 fairseq-train \
    /clwork/daisuke/100knock/data/jesc_data \
    --fp16 \
    --save-dir /clwork/daisuke/100knock/data/knock98 --max-epoch 5 --arch lstm --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric > knock98.log