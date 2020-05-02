from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn import metrics
import numpy

data_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
data_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42) 
y_train = data_train.target
y_test = data_test.target

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
X_train = vectorizer.fit_transform(data_train.data)
X_test = vectorizer.transform(data_test.data)
feature_names = numpy.asarray(vectorizer.get_feature_names())

print(X_train.shape, X_test.shape, feature_names)

clasifier = LinearSVC(penalty='l2', dual=False, tol=1e-3)
clasifier.fit(X_train, y_train)
pred = clasifier.predict(X_test)
print(metrics.accuracy_score(y_test, pred))
