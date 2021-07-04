#損失と正解率を計算する関数
def calculate_loss_and_acc(model, dataset, device, loss_fun):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for data in dataloader:
            inputs = data['inputs'].to(device)
            labels = data['labels'].to(device)
            outputs = model(inputs)

            loss += loss_fun(outputs, labels).item()

            pred = torch.argmax(outputs, dim=-1)
            total += len(inputs)
            correct += (pred==labels).sum().item()
    return loss/len(dataset) , correct/total


from torch.utils.data import DataLoader
import time
from torch import optim

#学習を行う関数train_model
def train_model(dataset_train, dataset_valid, batch_size, model, loss_fun, optimier, \
                num_epochs, collate_fn=None, device=None):
    model.to(device)

    #dataloader
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # スケジューラの設定
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=1e-5, last_epoch=-1)

    #学習
    log_train = []
    log_valid = []
    for epoch in range(num_epochs):
        #各エポックにかかる時間を測る
        start = time.time()

        model.train()
        for data in dataloader_train:
            #勾配を初期化
            optimizer.zero_grad()

            inputs = data['inputs'].to(device)
            labels = data['labels'].to(device)
            outputs = model(inputs)
            loss = loss_fun(outputs, labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        loss_train, acc_train = calculate_loss_and_acc(model, dataset_train, device, loss_fun)
        loss_valid, acc_valid = calculate_loss_and_acc(model, dataset_valid, device, loss_fun)
        log_train.append([loss_train, acc_train])
        log_valid.append([loss_valid, acc_valid])

        # チェックポイントの保存
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), \
                    'optimizer_state_dict': optimizer.state_dict()}, f'checkpoint{epoch + 1}.pt')
        
        end = time.time()

        # ログを出力
        print(f'epoch: {epoch + 1}, loss_train: {loss_train:.4f}, accuracy_train: {acc_train:.4f},\
                loss_valid: {loss_valid:.4f}, accuracy_valid: {acc_valid:.4f}, {(end - start):.4f}sec')
        
        # 検証データの損失が3エポック連続で低下しなかった場合は学習終了
        if epoch > 2 and log_valid[epoch - 3][0] <= log_valid[epoch - 2][0] <= log_valid[epoch - 1][0] <= log_valid[epoch][0]:
            break
        
        #スケジューラを進める
    
    return {'train' : log_train, 'valid' : log_valid}
import numpy as np
from matplotlib import pyplot as plt

def visualize_logs(log):
  fig, ax = plt.subplots(1, 2, figsize=(15, 5))
  ax[0].plot(np.array(log['train']).T[0], label='train')
  ax[0].plot(np.array(log['valid']).T[0], label='valid')
  ax[0].set_xlabel('epoch')
  ax[0].set_ylabel('loss')
  ax[0].legend()
  ax[1].plot(np.array(log['train']).T[1], label='train')
  ax[1].plot(np.array(log['valid']).T[1], label='valid')
  ax[1].set_xlabel('epoch')
  ax[1].set_ylabel('accuracy')
  ax[1].legend()
  plt.show()

#データ準備
dataset_train = MyDataset(X_train, y_train, tokenizer, word2id)
dataset_valid = MyDataset(X_valid, y_valid, tokenizer, word2id)
dataset_test = MyDataset(X_test, y_test, tokenizer, word2id)

model = MyRNN(len(set(word2id.values()))+1, 300, 50, 4)

loss_fun = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

log = train_model(dataset_train, dataset_valid, 1, model, loss_fun, optimizer, 10)

visualize_logs(log)
_, acc_train = calculate_loss_and_acc(model, dataset_train, None, loss_fun)
_, acc_test = calculate_loss_and_acc(model, dataset_test, None, loss_fun)
print(f'正解率（学習データ）：{acc_train:.3f}')
print(f'正解率（評価データ）：{acc_test:.3f}')