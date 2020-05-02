from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn import metrics
import numpy
import json
import sys
import random

proposals = open('data/parsed_data.json', 'r')
data = json.loads(proposals.read())
proposals.close()
random.shuffle(data)
data_train = [x['text'] for x in data[0:150]] 
y_train = [x['category'] for x in data[0:150]]

random.shuffle(data)
data_test = [x['text'] for x in data[150:300]] 
y_test = [x['category'] for x in data[150:300]]
print(y_train[0:5])
print(y_test[0:5])

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
X_train = vectorizer.fit_transform(data_train)
X_test = vectorizer.transform(data_test)
feature_names = numpy.asarray(vectorizer.get_feature_names())

# print(X_train.shape, X_test.shape, feature_names)

clasifier = LinearSVC(penalty='l2', dual=False, tol=1e-3)
# clasifier = PassiveAggressiveClassifier(max_iter=50)
clasifier.fit(X_train, y_train)
pred = clasifier.predict(X_test)
print(metrics.accuracy_score(y_test, pred))
print(pred[0:5])