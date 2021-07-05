!fairseq-preprocess --source-lang ja --target-lang en \
    --trainpref /content/drive/MyDrive/Dataset/kyoto-train.cln --validpref /content/drive/MyDrive/Dataset/kyoto-dev --testpref /content/drive/MyDrive/Dataset/kyoto-test \
    --thresholdsrc 3 \
    --thresholdtgt 3 \
    --destdir /content/drive/MyDrive/Dataset/bpe-OOM --bpe subword_nmt --workers 20

!CUDA_VISIBLE_DEVICES=0 fairseq-train \
    /content/drive/MyDrive/Dataset/bpe-OOM \
    --save-dir /content/drive/MyDrive/Dataset/checkpoints/transformer/ \
    --fp16 \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 --max-epoch 5 --update-freq 8\
    --eval-bleu --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --tensorboard-logdir /content/drive/MyDrive/Dataset/Log/tensor.log

#bleu 2.63 --> threshold src 3が影響？
#LSTM beamsearch 13.06のほうが高かった
