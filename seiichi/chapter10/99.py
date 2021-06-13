import flask, os
from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/translate', methods=['GET'])
def translate():
    response = {
        "Content-Type": "application/json"
    }
    # ensure an feature was properly uploaded to our endpoint
    if flask.request.method == "GET":
        if flask.request.args.get("src"):
            # read feature from json
            src = flask.request.args.get("src")
            os.system("echo {} | mecab -Owakati > ./work/src".format(src))
            os.system("fairseq-interactive --path save91/checkpoint10.pt data91 < ./work/src | grep '^H' | cut -f3 > ./work/tgt")
            with open("./work/tgt", "r") as f:
                tgt = f.readline().strip()
            response["tgt"] = tgt

    # return the data dictionary as a JSON response
    return flask.jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)

"""
$ python3 99.py
 * Serving Flask app "99" (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: on
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
 * Restarting with fsevents reloader
 * Debugger is active!
 * Debugger PIN: 303-817-002
127.0.0.1 - - [12/Jun/2021 16:46:35] "GET /translate?src=冴えない彼女の育てかた HTTP/1.1" 200 -

$ python3
>>> import requests
>>> res = requests.get("http://127.0.0.1:5000/translate?src=冴えない彼女の育てかた")
>>> res.text
'{\n  "Content-Type": "application/json", \n  "tgt": "His mother was a daughter of <unk> <unk> ."\n}\n'

"""