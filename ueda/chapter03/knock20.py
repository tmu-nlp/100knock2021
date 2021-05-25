import json
import gzip

def load_json():
    with gzip.open('c:\Git\jawiki-country.json.gz') as f:
        for line in f:
            data = json.loads(line)
            if data['title'] == "イギリス":
                return data['text']

if __name__ == "__main__":
    print(load_json())