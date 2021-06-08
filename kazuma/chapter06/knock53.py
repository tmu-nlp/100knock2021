from knock52 import train
def predict_titles(titles):
    lr_model, data, vectorizer = train()
    print(lr_model.predict(vectorizer.transform(titles).toarray()))
    print([max(i) for i in lr_model.predict_proba(vectorizer.transform(titles).toarray())])

if __name__ == "__main__":
    titles = ["US STOCKS-Futures dip ahead of jobs, GDP data", "Predict confidence scores for samples."]
    predict_titles(titles)
