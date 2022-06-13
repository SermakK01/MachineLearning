from sklearn import datasets
import pandas as pd

data_breast_cancer = datasets.load_breast_cancer(as_frame=True)

from sklearn.model_selection import train_test_split
X = data_breast_cancer.data[['mean texture', 'mean symmetry']]
y = data_breast_cancer.target
cancer_X_train, cancer_X_test, cancer_y_train, cancer_y_test= train_test_split(X,y, test_size = 0.2)

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

#tree
tree_clf = DecisionTreeClassifier()
tree_clf.fit(cancer_X_train, cancer_y_train)

#logic regression gitez
log_reg = LogisticRegression(solver="lbfgs")
log_reg.fit(cancer_X_train, cancer_y_train)

#knn
from sklearn.neighbors import KNeighborsClassifier
knn_reg = KNeighborsClassifier()
knn_reg.fit(cancer_X_train, cancer_y_train)

voting_clf_hard = VotingClassifier(
    estimators=[('lr', log_reg),
                ('tr', tree_clf),
                ('knn', knn_reg)],voting='hard')

voting_clf_hard.fit(cancer_X_train,cancer_y_train)
print(voting_clf_hard.predict(cancer_X_test))

from sklearn.metrics import accuracy_score
for clf in (log_reg, tree_clf, knn_reg, voting_clf_hard):
    clf.fit(cancer_X_train, cancer_y_train)
    cancer_y_pred = clf.predict(cancer_X_test)
    print(clf.__class__.__name__,
          accuracy_score(cancer_y_test, cancer_y_pred))


voting_clf_soft = VotingClassifier(
    estimators=[('lr', log_reg),
                ('tr', tree_clf),
                ('knn', knn_reg)],voting='soft')


voting_clf_soft.fit(cancer_X_train,cancer_y_train)
print(voting_clf_soft.predict(cancer_X_test))


results = []

from sklearn.metrics import accuracy_score
for clf in (tree_clf, log_reg, knn_reg, voting_clf_hard,voting_clf_soft):
    clf.fit(cancer_X_train, cancer_y_train)
    cancer_y_pred = clf.predict(cancer_X_test)
    cancer_y_pred_train = clf.predict(cancer_X_train)
    results.append([accuracy_score(cancer_y_train, cancer_y_pred_train),accuracy_score(cancer_y_test, cancer_y_pred)])
    print(clf.__class__.__name__,
          accuracy_score(cancer_y_test, cancer_y_pred))


print(results)

import pickle

with open('acc_vote.pkl', 'wb') as f:
    pickle.dump(results,f)

#TO DO

results_class = [(tree_clf),(log_reg),(knn_reg),(voting_clf_hard),(voting_clf_soft)]

with open('vote.pkl', 'wb') as f:
    pickle.dump(results_class,f)

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

#bagging basic
bag_clf_basic = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30, bootstrap=True)
#bag_clf_basic.fit(cancer_X_train, cancer_y_train)

#bagging 50%
bag_clf_50 = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30,max_samples=0.5, bootstrap=True)
#bag_clf_50.fit(cancer_X_train, cancer_y_train)

#pasting
past_clf_basic = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30, bootstrap=False)
#past_clf_basic.fit(cancer_X_train, cancer_y_train)

#bagging 50%
past_clf_50 = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30,max_samples=0.5, bootstrap=False)
#past_clf_50.fit(cancer_X_train, cancer_y_train)

#RandomForrest
rnd_clf = RandomForestClassifier(n_estimators=30)
#rnd_clf.fit(cancer_X_train, cancer_y_train)

#AdaBoost
ada_clf = AdaBoostClassifier(n_estimators=30)
#ada_clf.fit(cancer_X_train, cancer_y_train)

#GradientBoost
gbrt_clf = GradientBoostingClassifier(n_estimators = 30)
#gbrt_clf.fit(cancer_X_train, cancer_y_train)


result1 = []

for clf in (bag_clf_basic, bag_clf_50, past_clf_basic, past_clf_50, rnd_clf, ada_clf, gbrt_clf):
    clf.fit(cancer_X_train, cancer_y_train)
    cancer_y_pred = clf.predict(cancer_X_test)
    cancer_y_pred_train = clf.predict(cancer_X_train)
    result1.append([accuracy_score(cancer_y_train, cancer_y_pred_train), accuracy_score(cancer_y_test, cancer_y_pred)])


print(result1)

with open('acc_bag.pkl', 'wb') as f:
    pickle.dump(result1,f)

result1_class = [(bag_clf_basic),(bag_clf_50),(past_clf_basic),(past_clf_50),(rnd_clf),(ada_clf),(gbrt_clf)]

with open('bag.pkl', 'wb') as f:
    pickle.dump(result1_class,f)


X1 = data_breast_cancer.data
y1 = data_breast_cancer.target
cancer_X_train1, cancer_X_test1, cancer_y_train1, cancer_y_test1 = train_test_split(X1,y1, test_size = 0.2)

sampling = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30, bootstrap=False, max_features=2, max_samples=0.5)

result2 = []

sampling.fit(cancer_X_train1, cancer_y_train1)
cancer_y_pred1 = sampling.predict(cancer_X_test1)
cancer_y_pred_train1 = sampling.predict(cancer_X_train1)
result2.append([accuracy_score(cancer_y_train1, cancer_y_pred_train1),accuracy_score(cancer_y_test1, cancer_y_pred1)])

print(result2)

with open('acc_fea.pkl', 'wb') as f:
    pickle.dump(result2,f)

result2_class = [(sampling)]

print(result2_class)

with open('fea.pkl', 'wb') as f:
    pickle.dump(result2_class,f)

clfs = [
    BaggingClassifier(DecisionTreeClassifier(), n_estimators=30,
                      bootstrap=True),
    BaggingClassifier(DecisionTreeClassifier(), n_estimators=30,
                      bootstrap=True, max_samples=0.5),

    BaggingClassifier(DecisionTreeClassifier(), n_estimators=30,
                      bootstrap=False),
    BaggingClassifier(DecisionTreeClassifier(), n_estimators=30,
                      bootstrap=False, max_samples=0.5),

    RandomForestClassifier(n_estimators=30),

    AdaBoostClassifier(n_estimators=30),

    GradientBoostingClassifier(n_estimators=30)
]

clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30,
                        bootstrap_features=False, max_features=2,
                        bootstrap=True, max_samples=0.5)
clf.fit(cancer_X_train, cancer_y_train)

rows = []

for estimator, features in zip(clf.estimators_, clf.estimators_features_):
    cols = [X.columns[i] for i in features]
    acc_score_train = accuracy_score(cancer_y_train, estimator.predict(cancer_X_train[cols]))
    acc_score_test = accuracy_score(cancer_y_test, estimator.predict(cancer_X_test[cols]))
    new_row = [acc_score_train, acc_score_train, cols]
    rows.append(new_row)

# %%
acc_fea_rank_pkl = pd.DataFrame(rows, columns=['acc_score_train', 'acc_score_test', 'features'])
acc_fea_rank_pkl.sort_values(by=['acc_score_test', 'acc_score_train'], inplace=True, ascending=False)

with open('acc_fea_rank.pkl', 'wb') as f:
    pickle.dump(acc_fea_rank_pkl, f)