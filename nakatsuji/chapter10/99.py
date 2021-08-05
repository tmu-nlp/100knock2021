from bottle import route, run, template, request
from datetime import datetime
import subprocess
import MeCab
import time


shell = "CUDA_VISIBLE_DEVICES={$N} PYTHONIOENCODING=utf-8 fairseq-interactive /work/michitaka/100knock/99/data/bin --path /work/michitaka/100knock/99/model/checkpoint_best.pt --beam 1"

@route('/translate')
def output():
    now = datetime.now()
    return template("knock99", text_inp="", text_res="")

tagger = MeCab.Tagger("-Owakati")

@route("/translate", method="POST")
def translate():
    proc = subprocess.Popen(shell, encoding='utf-8', stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
    input_text = request.forms.input_text
    input_ = tagger.parse(input_text)
    proc.stdin.write(input_)
    proc.stdin.close()
    res = proc.stdout.readlines()[-2].strip().split("\t")[-1]

    return template("knock99", text_inp=input_text, text_res=res)

run(host="localhost", port=8080, debug=True)
