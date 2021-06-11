#!/bin/bash

for N in `seq 1 10` ; do
    fairseq-score --sys 95.$N.out.spacy --ref test.spacy.en > 95.$N.score
done