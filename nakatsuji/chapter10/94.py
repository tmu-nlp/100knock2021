'''
for N in `seq 1 20` ; do
    CUDA_VISIBLE_DEVICES=0 fairseq-interactive --path save91/checkpoint10.pt --beam $N data91 < tok/kyoto-test.ja | grep '^H' | cut -f3 > checkpoints/94.$N.out
done

for N in `seq 1 20` ; do
    fairseq-score --sys 94.$N.out --ref tok/kyoto-test.en > checkpoints/94.$N.score
done
'''