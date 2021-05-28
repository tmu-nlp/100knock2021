import json
import gzip

# filename = './data/jawiki-country.json'
zip_filename = './data/jawiki-country.json.gz'

# with open(filename, 'r') as f:
# 	obj = f.read()

obj_zip = []
with gzip.open(zip_filename, 'r') as f:
	for line in f:
		obj_zip.append(json.loads(line))

for i in obj_zip:
	if i['title'] == 'イギリス':
		print(i['text'])
