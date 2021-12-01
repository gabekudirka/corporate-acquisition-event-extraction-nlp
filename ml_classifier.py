import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import pickle

data = pd.read_csv('./output_100d_real.csv')

column_names = ['label','frequency_in_doc','sentence_loc', 'sentence_num']

for i in range(36):
    column_names.append('dep_'+str(i))
for i in range(15):
    column_names.append('head_pos_'+str(i))
for i in range(36):
    column_names.append('head_dep_'+str(i))
for i in range(8):
    column_names.append('entity_label_'+str(i))

for i in range(100):
    column_names.append('text_'+str(i))
for i in range(100):
    column_names.append('head_text_'+str(i))
for i in range(100):
    column_names.append('next_verb_'+str(i))
#data.columns = column_names

for i in range(100):
    column_names.append('two_prior_'+str(i))
for i in range(100):
    column_names.append('one_prior_'+str(i))
for i in range(100):
    column_names.append('one_post_'+str(i))
for i in range(100):
    column_names.append('two_post_'+str(i))

data.columns = column_names

data_labeled = data[data['label'] != 'none']
data_unlabeled = data[data['label'] == 'none']
print(data_unlabeled.shape)
print(data_labeled.shape)
balanced_data = data_labeled.append(data_unlabeled.sample(n=data_labeled.shape[0], random_state=1))
print(balanced_data.shape)
data = balanced_data

data['label'] = data['label'].astype('category')
categories = dict(enumerate(data['label'].cat.categories))
data['label'] = data['label'].cat.codes

X = data.drop('label', axis=1)
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

classifier = GradientBoostingClassifier(learning_rate=0.06, n_estimators=120, max_depth=4, max_features='sqrt', random_state=0, min_samples_split=50, min_samples_leaf=1)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print(categories) 
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

filename = 'model_1.sav'
pickle.dump(classifier, open(filename, 'wb'))