from knock52 import train_logistic
from sklearn.linear_model import LogisticRegression
import pandas as pd

def pred_cat(input):
    lr = train_logistic()
    return lr.predict_proba(input), lr.predict(input)