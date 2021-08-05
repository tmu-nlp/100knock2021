#LSTMに変更
'''
CUDA_VISIBLE_DEVICES=0 fairseq-train data98_kftt_baseline \
    --fp16 \
    --save-dir save98_kftt_baseline \
    --max-epoch 10 \
    --arch lstm --share-decoder-input-output-embed \
    --optimizer adam --clip-norm 1.0 \
    --lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 2000 \
    --dropout 0.1 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 8000 > 98_kftt.log
'''