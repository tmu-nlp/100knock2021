!fairseq-preprocess --source-lang ja --target-lang en \
    --trainpref /content/drive/MyDrive/Dataset/kyoto-train.cln --validpref /content/drive/MyDrive/Dataset/kyoto-dev --testpref /content/drive/MyDrive/Dataset/kyoto-test \
    --destdir /content/drive/MyDrive/Dataset/bpe --bpe subword_nmt --workers 20

!mkdir /content/drive/MyDrive/Dataset/checkpoints
!mkdir /content/drive/MyDrive/Dataset/checkpoints/bpe
!CUDA_VISIBLE_DEVICES=0 fairseq-train \
    /content/drive/MyDrive/Dataset/bpe \
    --save-dir /content/drive/MyDrive/Dataset/checkpoints/bpe/ \
    --bpe subword_nmt \
    --arch lstm --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 --max-epoch 5 \
    --eval-bleu --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric

!CUDA_VISIBLE_DEVICES=0 fairseq-interactive /content/drive/MyDrive/Dataset/bpe --path /content/drive/MyDrive/Dataset/checkpoints/bpe/checkpoint_best.pt

for N in tqdm(range(1, 100, 20)):
  !fairseq-generate /content/drive/MyDrive/Dataset/bpe  \
    --path /content/drive/MyDrive/Dataset/checkpoints/bpe/checkpoint_best.pt \
    --beam $N --batch-size 128 --remove-bpe | tee /content/drive/MyDrive/Dataset/bpe/gen.out
  !grep  ^H /content/drive/MyDrive/Dataset/bpe/gen.out | cut -f3 > /content/drive/MyDrive/Dataset/bpe/gen.out.sys
  !grep  ^T /content/drive/MyDrive/Dataset/bpe/gen.out | cut -f2 > /content/drive/MyDrive/Dataset/bpe/gen.out.ref
  !fairseq-score --sys /content/drive/MyDrive/Dataset/bpe/gen.out.sys --ref /content/drive/MyDrive/Dataset/bpe/gen.out.ref | tail -n1  >> /content/drive/MyDrive/Dataset/bleuscore.txt

  #BLEU4 = 12.74, 39.5/16.4/8.4/4.8 (BP=1.000, ratio=1.058, syslen=28280, reflen=26734)