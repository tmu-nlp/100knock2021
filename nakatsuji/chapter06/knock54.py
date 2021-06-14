from sklearn.metrics import accuracy_score
train_accuracy = accuracy_score(Y_train, train_pred[1])
test_accuracy = accuracy_score(Y_test, test_pred[1])
print(f'train : {train_accuracy}')
print(f'test : {test_accuracy}')