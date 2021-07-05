!pip install flask-ngrok
!pip install Flask
!pip install wtforms
# 形態素分析ライブラリーMeCab と 辞書(mecab-ipadic-NEologd)のインストール 
!apt-get -q -y install sudo file mecab libmecab-dev mecab-ipadic-utf8 git curl python-mecab > /dev/null
!git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git > /dev/null 
!echo yes | mecab-ipadic-neologd/bin/install-mecab-ipadic-neologd -n > /dev/null 2>&1
!pip install mecab-python3 > /dev/null

# シンボリックリンクによるエラー回避
!ln -s /etc/mecabrc /usr/local/etc/mecabrc

from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
from wtforms import Form, TextAreaField, validators
import subprocess
import MeCab

tagger = MeCab.Tagger("-Owakati")
app = Flask(__name__, template_folder='/content/drive/MyDrive/Dataset/templates')
run_with_ngrok(app)

class TransForm(Form):
    inputsent = TextAreaField('',[validators.DataRequired()])

@app.route('/')
def index():
    form = TransForm(request.form)
    return render_template('form.html', form=form)

@app.route('/trans', methods=['POST'])
def trans():
    form = TransForm(request.form)
    if request.method == 'POST' and form.validate():
        sent = request.form['enter sentence']
        sent = tagger.parse(sent)
        cp = subprocess.run(['fairseq-interactive', '--cpu', '/content/drive/MyDrive/Dataset/bpe/ ', '--path', '/content/drive/MyDrive/Dataset/checkpoints/bpe/ '], input=name, encoding='UTF-8', stdout=subprocess.PIPE)
        name = cp.stdout.split("\n")[-3].split("\t")[2]
        return render_template('results.html', input_sent=sent, output_sent=name)
    return render_template('form.html', form=form)

if __name__ == '__main__':
    app.run()