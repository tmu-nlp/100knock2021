# 99. 翻訳サーバの構築
# ユーザが翻訳したい文を入力すると，
# その翻訳結果がウェブブラウザ上で表示されるデモシステムを構築せよ．

import time
import MeCab
import subprocess
from bottle import route, run, template, request

BIN = '/work/aomi/100knock2020/chapter10/data/processed_16000/bin'
MODEL = '/work/aomi/100knock2020/chapter10/knock95/models/model_1111/checkpoint_best.pt'
SHELL = f'CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 fairseq-interactive {BIN} --path {MODEL}'

tagger = MeCab.Tagger('-Owakati')

@route('/translate')
def output():
    return template('knock99', text_inp'', text_res = '')

@route('/translate', method = 'POST')
def translate():
    proc = subprocess.Popen(SHELL, encoding = 'utf-8', stdin = subprocess.PIPE, stdout = subprocess.PIPE, shell = True)
    text = request.forms.input_text
    tok_text = tagger.parse(text)
    proc.stdin.write(tok_text)
    proc.stdin.close()
    res = proc.stdout.readlines()[-2].strip().split('\t')[-1]
    return template('knock99', text_inp = text, text_res = res)

if __name__ == '__main__':
    run(host = 'localhost', port = 8080, debug = True)
