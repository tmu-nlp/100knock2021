from flask import Flask, request, render_template
from wtforms import Form, TextAreaField, validators
import subprocess
import MeCab

tagger = MeCab.Tagger("-Owakati")

app = Flask(__name__)

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
        input = request.form['inputsent']
        input = tagger.parse(input)
        cp = subprocess.run(['fairseq-interactive', '--cpu', 'data-bin/kftt.ja-en', '--path', 'checkpoints/kftt.ja-en/checkpoint_best.pt'], input=input, encoding='UTF-8', stdout=subprocess.PIPE)
        output = cp.stdout.split("\n")[-3].split("\t")[2]
        return render_template('trans.html', output=output)
    return render_template('form.html', form=form)

if __name__ == '__main__':
    app.run()