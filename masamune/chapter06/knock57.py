import joblib

model = joblib.load('model.joblib')
vocabulary = joblib.load('vocabulary.joblib')

#特徴量の重み
coefs = model.coef_

category_names = ["business", "entertainment", "health", "science and technology"]

for i, category_features in enumerate(coefs):
    features = dict()
    for word, index in vocabulary.items():
        features[word] = category_features[index]

    print(f'Top 10 of {category_names[i]}')
    for word, weight in sorted(features.items(), key=lambda x:x[1], reverse=True)[:10]:
        print(f"{word} -> {weight}")
    print()

    # 重みの低い特徴量トップ10を表示する
    print(f'Worst 10 of {category_names[i]}')
    for word, weight in sorted(features.items(), key=lambda x:x[1], reverse=False)[:10]:
        print(f"{word} -> {weight}")
    print()