path = '/content/drive/MyDrive/nlp100/'
filename = path + 'vocabulary.sav'
vocabulary = pickle.load(open(filename, 'rb'))


categories = ["business", "entertainment", "health", "science and technology"]
for i, c_features in enumerate(loaded_model.coef_):
    features = dict()
    for word, idx in vocabulary.items():
        features[word] = c_features[idx]
    features = sorted(features.items(), key=lambda x:x[1], reverse=True)
    #高い順
    print(categories[i])
    print('高い順')
    for feature in features[:10]:
        print(f'{feature[0]} : {feature[1]}')
    print()
    #低い順
    print('低い順')
    features.reverse()
    for feature in features[:10]:
        print(f'{feature[0]} : {feature[1]}')
    print()