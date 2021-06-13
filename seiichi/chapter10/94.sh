#!/bin/bash

for N in `seq 1 20` ; do
    fairseq-interactive --path save91/checkpoint10.pt --beam $N data91 < test.spacy.ja | grep '^H' | cut -f3 > 94.$N.out
done

for N in `seq 1 20` ; do
    fairseq-score --sys 94.$N.out --ref test.spacy.en > 94.$N.score
done