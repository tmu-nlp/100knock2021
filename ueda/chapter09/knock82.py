#knock82.py
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
cat = str.maketrans({'b':'0', 't':'1', 'e':'2', 'm':'3'})

def create_dataset(file_name):
    with open(file_name, encoding="utf-8") as f:
        x_vec=[]
        y_vec=[]
        average = 0
        for line in f:
            y, sent = line.strip().split('\t')
            x_vec.append(sentence2ids(wordids, sent))
            y_vec.append(int(y.translate(cat)))
        x_vec = ids2tensor(x_vec, int(np.mean([len(a) for a in x_vec])))
        y_vec = torch.tensor(y_vec, dtype = torch.int64)
        return x_vec, y_vec

def ids2tensor(vec, max_length):
    x_vec = []
    for ids in vec:
        if len(ids) > max_length:
            ids = ids[:max_length]
        else:
            ids += [len(wordids)] * (max_length - len(ids))
        x_vec.append(ids)
    return torch.tensor(x_vec, dtype=torch.int64)

def create_data(x, y):
    data = []
    for i in range(len(y)):
        data.append([x[i], y[i]])
    return data

def calc_accuracy(pred, answer):
    correct = 0
    for i in range(len(answer)):
        if pred[i] == answer[i]: correct+=1
    return correct/len(answer)

model = RNN(len(wordids)+1, 300, 50, 4, len(wordids))
x_train, y_train = create_dataset(r'C:\Git\train.txt')
x_test, y_test = create_dataset(r'C:\Git\test.txt')
dataset = create_data(x_train, y_train)
train_loader = DataLoader(dataset, shuffle = True)
optim = torch.optim.SGD(model.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()

train_accuracy = []
test_accuracy = []
train_loss = []
test_loss = []
for epoch in tqdm(range(10)):
    for inputs, target in train_loader:
        optim.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, target)
        loss.backward()
        optim.step()
    with torch.no_grad():
        pred = model(x_train)
        train_accuracy.append(calc_accuracy(torch.argmax(pred, dim=1), y_train))
        train_loss.append(loss_func(pred, y_train))
        pred = model(x_test)
        test_accuracy.append(calc_accuracy(torch.argmax(pred, dim=1), y_test))
        test_loss.append(loss_func(pred, y_test))

plt.figure(figsize=(20, 10))
plt.subplot(1,2,1)
plt.plot(train_accuracy, label='train')
plt.plot(test_accuracy, label='test')
plt.subplot(1,2,2)
plt.plot(train_loss, label='train')
plt.plot(test_loss, label='test')
plt.show()