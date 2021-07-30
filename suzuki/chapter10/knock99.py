from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import subprocess

app = Flask(__name__)

class TransForm(Form):
    entersent = TextAreaField('',[validators.DataRequired()])

@app.route('/')
def index():
    form = TransForm(request.form)
    return render_template('app.html', form=form)

@app.route('/trans', methods=['POST'])
def trans():
    form = TransForm(request.form)
    if request.method == 'POST' and form.validate():
        name = request.form['entersent']
        cp = subprocess.run(['fairseq-interactive', '--cpu', 'data-bin/kftt.ja-en', '--path', 'save95/checkpoint_best.pt'], input=name, encoding='UTF-8', stdout=subprocess.PIPE)
        name = cp.stdout.split("\n")[-3].split("\t")[2]
        return render_template('translated.html', name=name)
    return render_template('app.html', form=form)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5800, threaded=True)