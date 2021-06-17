import random

def list_to_txt(filename, List):
    file = open(filename, 'w')
    for line in List:
        file.write(line + '\n')
    file.close()

if __name__ == '__main__':
    f = open('./NewsAggregatorDataset/newsCorpora.csv', 'r')
    ans = []

    #publisherが一致する情報源を抽出
    for i , line in enumerate(f):
        l = line.strip().split('\t')
        publisher = l[3]

        if publisher == 'Reuters' \
            or publisher == 'Huffington Post' \
            or publisher == 'Businessweek' \
            or  publisher == 'Contactmusic.com' \
            or  publisher == 'Daily Mail':

            ans.append(l[4] + '\t' + l[1])
    
    f.close()

    #情報源をランダムに並び替え
    random.shuffle(ans)

    print(len(ans)) #13356

    train = ans[:int(len(ans) * 0.8)]
    valid = ans[int(len(ans) * 0.8):int(len(ans) * 0.9)]
    test = ans[int(len(ans) * 0.9):]

    list_to_txt('train.txt', train)
    list_to_txt('valid.txt', valid)
    list_to_txt('test.txt', test)



'''

事例数の確認

10684   train.txt
1336    valid.txt
1336    test.txt
--------------------
13356   total

'''