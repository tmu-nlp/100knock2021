!fairseq-generate /content/drive/MyDrive/Dataset/kyoto  \
    --path /content/drive/MyDrive/Dataset/checkpoints/kyoto/checkpoint_best.pt \
    --beam 5 --batch-size 128 --remove-bpe | tee /content/drive/MyDrive/Dataset/checkpoints/kyoto/gen.out
!grep  ^H /content/drive/MyDrive/Dataset/kyoto/gen.out | cut -f3 > /content/drive/MyDrive/Dataset/kyoto/gen.out.sys
!grep  ^T /content/drive/MyDrive/Dataset/kyoto/gen.out | cut -f2 > /content/drive/MyDrive/Dataset/kyoto/gen.out.ref
!fairseq-score --sys /content/drive/MyDrive/Dataset/kyoto/gen.out.sys --ref /content/drive/MyDrive/Dataset/kyoto/gen.out.ref

#BLEU4 = 12.60, 39.4/16.0/8.4/4.8 (BP=1.000, ratio=1.061, syslen=28372, reflen=26734)