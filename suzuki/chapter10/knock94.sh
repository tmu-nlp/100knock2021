for N in `seq 1 100` ; do
    CUDA_VISIBLE_DEVICES=0 fairseq-generate /clwork/daisuke/100knock/data/data-bin/kftt.ja-en  \
        --path /clwork/daisuke/100knock/data/knock91/checkpoint_best.pt \
        --beam $N --batch-size 128 --remove-bpe | tee /tmp/gen.out

    grep ^H /tmp/gen.out | cut -f3- > /tmp/gen.out.sys
    grep ^T /tmp/gen.out | cut -f2- > /tmp/gen.out.ref

    fairseq-score --sys /tmp/gen.out.sys --ref /tmp/gen.out.ref >> knock94.score
done