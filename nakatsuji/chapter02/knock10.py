#from os import path
#path = path.join(path.dirname(__file__), 'popular-names.txt')
with open("NLP_100/chapter02/popular-names.txt", 'r') as f, open("NLP_100/chapter02/py/knock10_py.txt", "w") as Tof:
    length = len(f.readlines())
    Tof.write('{}'.format(length))