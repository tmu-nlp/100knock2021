TEXT=kftt-data-1.0/data/tok
fairseq-preprocess --source-lang ja --target-lang en \
    --trainpref $TEXT/kyoto-train --validpref $TEXT/kyoto-dev --testpref $TEXT/kyoto-test \
    --destdir data-bin/kftt.ja-en \
    --workers 20