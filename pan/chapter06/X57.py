#特徴量の重みの確認
import joblib

if __name__ == '__main__':
    clf = joblib.load('model.joblib')
    vocabulary_ = joblib.load('vocabulary_.joblib')

    # 特徴量の重みを得る
    # 各カテゴリごとに特徴量の重みが入っているというリストのリストになっている
    coefs = clf.coef_

    category_names = ['business', 'entertainment', 'health', 'science and technology']

    # 各カテゴリごとに重みの高い特徴量トップ10と、重みの低い特徴量トップ10を得る
    for i, category_features in enumerate(coefs):
        # keyを語彙、valueをその語彙の重みとする辞書を作る
        features = dict()
        for word, index in vocabulary_.items():
            features[word] = category_features[index]

        # 重みの高い特徴量トップ10を表示する
        print(f'Top 10 of \'{category_names[i]}\':')
        for word, weight in sorted(features.items(), key = lambda x:x[1], reverse = True)[:10]:
            print(f'{word:<10s} -> {weight}')
        print()

        # 重みの低い特徴量トップ10を表示する
        print(f'Worst 10 of \'{category_names[i]}\':')
        for word, weight in sorted(features.items(), key = lambda x:x[1], reverse = False)[:10]:
            print(f'{word:<10s} -> {weight}')
        print()