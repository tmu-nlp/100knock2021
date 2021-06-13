import pandas as pd
import joblib

test_feature_path = './data/test.feature.txt'

#データ読み込み
names = ['TITLE', 'CATEGORY']
X_test = pd.read_csv(test_feature_path, sep='\t', header=None)

#モデル読み込み
model = joblib.load('model.joblib')

#予測
Y_pred = model.predict(X_test)
Y_prob = model.predict_proba(X_test)

#予測カテゴリと予測確率
for i in range(5):
    print(f"predict:{Y_pred[i]}\tb:{Y_prob[i][0]} e:{Y_prob[i][1]} m:{Y_prob[i][2]} t:{Y_prob[i][3]}")