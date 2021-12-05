import pandas as pd
import numpy as np
import pickle
from SubjectProblemList import getProblems, getSubjects
from numpy import format_parser

from Code.Approach2.Missing_Students import find_missing_students

missing = find_missing_students(True)

late_train = pd.read_csv('../../data/S19/Train/late.csv')
last_predictions = []

with open('../Approach2/LateTrainAllFeatures.pickle', 'rb') as handle:
    X_Train = pickle.load(handle)


# print(len(X_train))
TrainLateProblemList = pd.read_csv('../LateTrainProblemList.csv')


def perProblemFeatures(X_train, prob_id):
    # print('Problem ID: ', prob_id)
    n_previously_generated_features = 8
    subjectFeatures = []
    for i in range(len(X_train)):
        if X_train[i, n_previously_generated_features] == prob_id:
            subjectFeatures.append(X_train[i, :8])  # Try with different features from 5-8 (First 5 from Naive Model)
    return subjectFeatures


problem_X_Train = []
sum = 0

for i in range(1):
    # print(TrainLateProblemList['ProblemID'].iloc[i])
    problem_X_Train.append(perProblemFeatures(X_Train, TrainLateProblemList['ProblemID'].iloc[i]))
    sum += len(problem_X_Train[i])
    # print(len(problem_X_Train[i]))

y_Train = []

# print('X_Train:', len(problem_X_Train[3]))
# print(sum)
# print(len(y_Train))
# print(len(y_Train[3]))

## Making 80_20
from sklearn.model_selection import train_test_split

macrof1_model = []
macrof1_cv = []
AUC_model = []
AUC_cv = []

for i in range(20):
    X_train, X_test, Y_train, Y_test = train_test_split(problem_X_Train[i], y_Train[i], test_size=0.2, shuffle=False)
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.svm import LinearSVC

    # model = LogisticRegressionCV()
    model = RandomForestClassifier(n_estimators=500, max_leaf_nodes=64, n_jobs=-1)
    # rbf_clf = SVC(kernel='rbf', gamma=5, C=0.001)
    # rbf_clf.fit(X_train, Y_train)
    # train_predictions = rbf_clf.predict(X_test)

    # svmPoly = sklearn.svm.SVC(kernel='poly', C=0.05, degree=4, coef0=1, gamma=1)
    # svmPoly.fit(X_train,Y_train)
    # linear_clf = LinearSVC(C=1, loss='hinge', random_state=42)
    # linear_clf.fit(X_train, Y_train)
    # train_predictions = linear_clf.predict(X_test)
    # train_predictions=svmPoly.predict(X_test)
    # clf = MLPClassifier(solver='lbfgs', alpha=1e-7, hidden_layer_sizes=(100, 50, 2), max_iter=3000, random_state=42)
    # model = BaggingClassifier(base_estimator=clf, n_estimators=500, random_state=1)

    if i != 0:
        # lastColumn='LastPredictions'+str(i+30)
        X_train = np.asarray(X_train)
        X_test = np.asarray(X_test)
        # print(type(X_train))type
        # print(type(X_train[0]))

        X_train = np.append(X_train, [[np.NaN]] * len(X_train), axis=1)
        X_train = np.append(X_train, [column[i] for column in missing], axis=1)
        # X_test = np.append(X_test, [column[i] for column in missing], axis=1)
        X_test = np.append(X_test, [[np.NaN]] * len(X_test), axis=1)

        print(len(X_train))
        break
        ind_pred = 0
        problem_list = getProblems(late_train)
        n_previously_generated_features = 8
        for k in range(len(X_train)):
            for b in range(len(missing)):
                if missing[b][i] == True:
                    if X_train[k, n_previously_generated_features] == problem_list[i]:
                        X_train[k, -1] = last_predictions[ind_pred]
                        ind_pred += 1
                else:
                    X_train[k, -1] = np.median(last_predictions)

        for k in range(len(X_test)):
            for b in range(len(missing)):
                if missing[b][i] == True:
                    if X_test[k, n_previously_generated_features] == problem_list[i]:
                        X_test[k, -1] = last_predictions[ind_pred]
                        ind_pred += 1
                else:
                    X_test[k, -1] = np.median(last_predictions)

    model.fit(X_train, Y_train)
    train_predictions = model.predict(X_test)
    last_predictions = train_predictions

    from sklearn.metrics import classification_report
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import f1_score

    print(classification_report(Y_test, train_predictions))
    print('AUC: ' + str(roc_auc_score(Y_test, train_predictions)))
    print('Macro F1: ' + str(f1_score(Y_test, train_predictions, average='macro')))
    macrof1_model.append((f1_score(Y_test, train_predictions, average='macro')))
    AUC_model.append(roc_auc_score(Y_test, train_predictions))
    from sklearn.metrics import plot_roc_curve

    # plot_roc_curve(model, X_train, y_train)

    from sklearn.model_selection import cross_validate

    # model = LogisticRegressionCV()
    model = RandomForestClassifier(n_estimators=500, max_leaf_nodes=64, n_jobs=-1)
    # model = BaggingClassifier(base_estimator=clf, n_estimators=65, random_state=42)
    cv_results = cross_validate(model, X_train, Y_train, cv=10, scoring=['accuracy', 'f1_macro', 'roc_auc'])
    # cv_results = cross_validate(rbf_clf, X_train, Y_train, cv=10, scoring=['accuracy', 'f1_macro', 'roc_auc'])
    # cv_results = cross_validate(svmPoly, X_train, Y_train, cv=10, scoring=['accuracy', 'f1_macro', 'roc_auc'])
    print(f'Accuracy: {np.mean(cv_results["test_accuracy"])}')
    print(f'AUC: {np.mean(cv_results["test_roc_auc"])}')
    print(f'Macro F1: {np.mean(cv_results["test_f1_macro"])}')
    macrof1_cv.append(np.mean(cv_results["test_f1_macro"]))
    AUC_cv.append(np.mean(cv_results["test_roc_auc"]))

print('Mean model f1 score: ', np.mean(macrof1_model))
print("Mean cv f1 score: ", np.mean(macrof1_cv))
print('AUC Model: ', np.mean(AUC_model))
print('AUC CV: ', np.mean(AUC_cv))
