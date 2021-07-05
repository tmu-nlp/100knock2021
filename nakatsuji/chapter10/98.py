
'''
CUDA_VISIBLE_DEVICES=1 fairseq-train data98_kftt \
    --fp16 \
    --restore-file save98/checkpoint3.pt \
    --save-dir save98_kftt \
    --max-epoch 10 \
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --clip-norm 1.0 \
    --lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 2000 \
    --dropout 0.1 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 8000 > 98_kftt.log
'''