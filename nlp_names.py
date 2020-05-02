import nltk
from nltk.corpus import names
import random

def gender_features(word):
    return{'last_letter': word[-1]}

def gender_features2(word):
    return{'suffix1': word[-1], 'suffix2': word[-2]}

def gender_features_own(word):
    features = {}
    name = word.lower()
    features['last_letter'] = name[-1]
    features['len'] = len(name)
    features['first_letter'] = name[0]

    return features

def gender_features_extended(word):
    features = {}
    name = word.lower()
    features['first_letter'] = name[0]
    features['last_letter'] = name[-1]
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features["count({})".format(letter)] = name.count(letter)
        features["has({})".format(letter)] = (letter in name)

    return features


labeled_names = ([(name, 'male') for name in names.words('male.txt')] + 
    [(name, 'female') for name in names.words('female.txt')])
random.shuffle(labeled_names)


train_set = labeled_names[1500:]
dev_set = labeled_names[500:1500]
test_set = labeled_names[:500]

train_feature_set = [(gender_features2(n), gender) for (n, gender) in train_set]
dev_feature_set = [(gender_features2(n), gender) for (n, gender) in dev_set]
test_feature_set = [(gender_features2(n), gender) for (n, gender) in test_set]

classifier = nltk.NaiveBayesClassifier.train(train_feature_set)

errors = []
for (name, tag) in dev_set:
    guess = classifier.classify(gender_features2(name))
    if guess != tag:
        errors.append((tag, guess, name))

for (tag, guess, name) in sorted(errors):
    print('correct={:<8} guess={:<8s} name={:<30}'.format(tag, guess, name)) 

# print(classifier.show_most_informative_features(26))
print(nltk.classify.accuracy(classifier, dev_feature_set))