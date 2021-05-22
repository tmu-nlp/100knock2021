import json
import sys
json_open = open("jawiki-country.json", "r").readlines()
for line in json_open:
    now = json.loads(line)
    if now['title'] == 'イギリス':
        with open('data/uk-text.txt', 'w') as f:
            f.write(now['text'])
        text = now['text']

def print_text(text):
    print(text)

if __name__ == "__main__":
    print_text(text)