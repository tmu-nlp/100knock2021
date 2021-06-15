line = ['hello i am daisuke suzuki', 'hello i am john cage', 'i wanna quit lab', 'ninhao wo jao lingmu dayue']

for l in line:
    L = l.strip().split(' ')
    if 'hello' in L[0] or 'ninhao' in L[0]:
        print(L)
        print(L[3] + ' ' + L[4])