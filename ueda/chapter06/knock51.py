from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

vectorizer = TfidfVectorizer(min_df=5, ngram_range=(1, 2))

def create_features(file_name):
        with open('C:\Git\{}.txt'.format(file), encoding="utf-8") as f:
            if file_name == 'train':
                X = vectorizer.fit_transform(f)
            else:
                X = vectorizer.transform(f)
            X = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
            X.to_csv('C:\Git\{}.feature.txt'.format(file), sep=',', index=False)

files = ['train', 'valid', 'test']
for file in files:
    X = create_features(file)


