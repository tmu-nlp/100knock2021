from os import path
path = path.join(path.dirname(__file__), 'popular-names.txt')
with open("NLP_100/chapter02/popular-names.txt", 'r') as f, open('NLP_100/chapter02/py/knock11_py.txt', 'w') as Tof:
    for line in f.readlines():
        Tof.write(line.replace('\t', ' '))
    