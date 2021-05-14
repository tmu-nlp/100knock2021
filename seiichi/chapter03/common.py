import json
def load_json(title='イギリス'):
    tar = open("./jawiki-country.json", "r").readlines()
    for line in tar:
        j = json.loads(line)
        if j['title'] == title:
            return j['text']
if __name__ == "__main__":
    ret = load_json()
    print(ret)
