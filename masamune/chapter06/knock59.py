from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
from tqdm import tqdm

#データ読み込み
features = ['TITLE', 'CATEGORY']
train_df = pd.read_csv('./data/train.txt', sep='\t', header=None, names=features)
valid_df = pd.read_csv('./data/valid.txt', sep='\t', header=None, names=features)
test_df = pd.read_csv('./data/test.txt', sep='\t', header=None, names=features)

X_train = pd.read_csv('./data/train.feature.txt', sep='\t', header=None)
Y_train = train_df['CATEGORY']
X_valid = pd.read_csv('./data/valid.feature.txt', sep='\t', header=None)
Y_valid = valid_df['CATEGORY']
X_test = pd.read_csv('./data/test.feature.txt', sep='\t', header=None)
Y_test = test_df['CATEGORY']

#標準化
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_valid = sc.transform(X_valid)
X_test = sc.transform(X_test)

#更新用の変数
best_model = None
best_acc = 0

#正解率とモデルを更新するか判断する関数
def judge(model):
    model.fit(X_train, Y_train)
    valid_acc = accuracy_score(Y_valid, model.predict(X_valid))
    global best_acc
    if best_acc < valid_acc:
        best_acc = valid_acc
        global best_model
        best_model = model

#ロジスティック回帰
penalties = ['l1', 'l2'] #正則化
params = [0.01, 0.1, 1.0] #正則化項のハイパーパラメータ
for penalty in tqdm(penalties):
    for C in tqdm(params):
        model = LogisticRegression(penalty=penalty, solver='liblinear', random_state=0, C=C, max_iter=1000)
        judge(model)
test_acc = accuracy_score(Y_test, best_model.predict(X_test))
print(f'test_acc: {test_acc}')
print(best_model)
best_acc = 0

'''
#SVM
params = [0.01, 0.1, 1.0]
for C in tqdm(params):
    model = SVC(kernel="linear", random_state=0, C=C)
    judge(model)
test_acc = accuracy_score(Y_test, best_model.predict(X_test))
print(f'test_acc: {test_acc}')
print(best_model)
best_acc = 0
'''

#Random Forest
depths = [128, 256, 512, 1024, 2048] #決定木の深さ
for depth in depths:
    model = RandomForestClassifier(max_depth=depth, random_state=0)
    judge(model)
test_acc = accuracy_score(Y_test, best_model.predict(X_test))
print(f'test_acc: {test_acc}')
print(best_model)
best_acc = 0

#K近傍法
k_range = range(1, 31) #最近傍数
for k in tqdm(k_range):
    model = KNeighborsClassifier(n_neighbors=k)
    judge(model)
test_acc = accuracy_score(Y_test, best_model.predict(X_test))
print(f'test_acc: {test_acc}')
print(best_model)
best_acc = 0

#lightgbm
depths = range(-1, 4) #木の深さ
leaves = range(29, 34) #木を分割した後の葉の数
for depth in tqdm(depths):
    for leave in tqdm(leaves):
        model = lgb.LGBMClassifier(max_depth=depth, num_leaves=leave)
        judge(model)
test_acc = accuracy_score(Y_test, best_model.predict(X_test))
print(f'test_acc: {test_acc}')
print(best_model)

'''
結果
・ロジスティック回帰
test_acc: 0.9122938530734632
LogisticRegression(C=0.01, max_iter=1000, random_state=0, solver='liblinear')

・Random Forest
test_acc: 0.8035982008995503
RandomForestClassifier(max_depth=1024, random_state=0)

・k近傍法
test_acc: 0.7608695652173914
KNeighborsClassifier(n_neighbors=1)

・lightgbm
test_acc: 0.8343328335832084
LGBMClassifier(num_leaves=33)
'''