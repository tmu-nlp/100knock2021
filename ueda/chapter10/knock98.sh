!fairseq-preprocess --source-lang ja --target-lang en \
    --trainpref /content/drive/MyDrive/Dataset/JESC-train --validpref /content/drive/MyDrive/Dataset/kyoto-dev --testpref /content/drive/MyDrive/Dataset/kyoto-test \
    --destdir /content/drive/MyDrive/Dataset/JESC-bpe --bpe subword_nmt --workers 20

!mkdir /content/drive/MyDrive/Dataset/checkpoints/JESC-bpe
!CUDA_VISIBLE_DEVICES=0 fairseq-train \
    /content/drive/MyDrive/Dataset/JESC-bpe \
    --fp16 \
    --save-dir /content/drive/MyDrive/Dataset/checkpoints/JESC-bpe/ \
    --bpe subword_nmt \
    --arch lstm --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 --max-epoch 5 \
    --eval-bleu --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric