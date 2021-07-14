fairseq-interactive data-bin/kftt.ja-en/ \
    --path checkpoints/kftt.ja-en/checkpoint_best.pt \
    < ../kftt-data-1.0/data/tok/kyoto-test.ja \
    | grep '^H' | cut -f3 > 92.out

head ../kftt-data-1.0/data/tok/kyoto-test.ja
head 92.out