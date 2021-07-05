for N in tqdm(range(1, 100, 5)):
  !fairseq-generate /content/drive/MyDrive/Dataset/kyoto  \
    --path /content/drive/MyDrive/Dataset/checkpoints/kyoto/checkpoint_best.pt \
    --beam $N --batch-size 128 --remove-bpe | tee /content/drive/MyDrive/Dataset/kyoto/gen.out
  !grep  ^H /content/drive/MyDrive/Dataset/kyoto/gen.out | cut -f3 > /content/drive/MyDrive/Dataset/kyoto/gen.out.sys
  !grep  ^T /content/drive/MyDrive/Dataset/kyoto/gen.out | cut -f2 > /content/drive/MyDrive/Dataset/kyoto/gen.out.ref
  !fairseq-score --sys /content/drive/MyDrive/Dataset/kyoto/gen.out.sys --ref /content/drive/MyDrive/Dataset/kyoto/gen.out.ref | tail -n1  >> /content/drive/MyDrive/Dataset/bleuscore.txt