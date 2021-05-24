import json
import gzip

with gzip.open('jawiki-country.json.gz') as file, open('uk.json', 'w') as f:
    for line in file:
        line = json.loads(line)
        if line['title'] == 'イギリス':
            f.write(line['text'])
            break

with open('uk.json') as file:
    for line in file:
        line = line.replace('\n', '')
        print(line)