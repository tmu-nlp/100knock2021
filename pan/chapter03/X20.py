#Read the JSON documents and output the body of the article about the United Kingdom. Reuse the output in problems 21-29.
import gzip
import json

def read_json(file_path):
    with gzip.open(file_path,'rt') as f:
        for line in f:
            file_data = json.loads(line)
            if file_data['title'] == 'United Kingdom':
                return(file_data['text'])

if __name__ == '__main__':
    file_path = '/users/kcnco/github/100knock2021/pan/chapter03/enwiki-country.json.gz'
    print(read_json(file_path))

with open('/users/kcnco/github/100knock2021/pan/chapter03/x20.txt','w') as f_name:
    f_name.write(text)
