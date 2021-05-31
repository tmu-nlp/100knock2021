import json
import gzip

PATH = 'jawiki-country.json.gz'

def load_england():
    with gzip.open(PATH, mode='r') as f:
        for line in f.readlines():
            decoded = json.loads(line)
            if decoded['title'] == 'イギリス':
                return decoded['text']

if __name__ == '__main__':
    print(load_england())
