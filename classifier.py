import json
import trainer
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import numpy as np
from scipy.stats import sem

rs = json.load(open("results.xml"))
trn = trainer.Trainer()

for i in range(len(rs)):
    trn.results[i] = rs[u'%i' % i]


indices = [i for i in trn.results]
data = trainer.face_dataset.data[indices, :]

target = [trn.results[i] for i in trn.results]
target = np.array(target).astype(np.int32)

clf = svm.SVC(kernel='linear')
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=0)


def evaluate_cross_val(clf, X, y):
    cv = KFold(len(y), shuffle=True, random_state=0)
    scores = cross_val_score(clf, X, y, cv=cv)
    print scores
    print "Mean score: {0:.3f} (+/-{1:.3f})".format(np.mean(scores), sem(scores))


def train_evaluate(clf, x_train, x_test, y_train, y_test):
    clf.fit(x_train, y_train)

    print ("Accuracy on training set:")
    print (clf.score(x_train, y_train))
    print ("Accuracy on testing set:")
    print (clf.score(x_test, y_test))

    y_pred = clf.predict(x_test)

    print ("Classification Report:")
    print (metrics.classification_report(y_test, y_pred))
    print ("Confusion Matrix:")
    print (metrics.confusion_matrix(y_test, y_pred))


def predict(x_test):
    y_pred = clf.predict(x_test)
    return y_pred


train_evaluate(clf, x_train, x_test, y_train, y_test)
