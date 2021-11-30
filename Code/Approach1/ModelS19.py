import pandas as pd
import numpy as np
import pickle


late_train = pd.read_csv('S19.zip/data/Train/late.csv')


with open('Code/LateTrainAllFeatures.pickle', 'rb') as handle:
    X_Train = pickle.load(handle)

# print(len(X_train))
TrainLateProblemList = pd.read_csv('Code/LateTrainProblemList.csv')


def perProblemFeatures(X_train, prob_id):
    # print('Problem ID: ', prob_id)
    n_previously_generated_features = 8
    subjectFeatures = []
    for i in range(len(X_train)):
        if X_train[i, n_previously_generated_features] == prob_id:
            subjectFeatures.append(X_train[i, :]) # Try with different features from 5-8 (First 5 from Naive Model)
    return subjectFeatures


problem_X_Train = []
sum = 0

for i in range(20):
    # print(TrainLateProblemList['ProblemID'].iloc[i])
    problem_X_Train.append(perProblemFeatures(X_Train, TrainLateProblemList['ProblemID'].iloc[i]))
    sum += len(problem_X_Train[i])
    # print(len(problem_X_Train[i]))


y_Train = []

for i in range(len(TrainLateProblemList)):
    y_TrainPerProblem = []
    for j in range(len(late_train)):
        if late_train['ProblemID'].iloc[j] == TrainLateProblemList['ProblemID'].iloc[i]:
            y_TrainPerProblem.append(late_train['Label'].iloc[j])
    y_Train.append(y_TrainPerProblem)

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

for l in range(20):
    X_train, X_test, Y_train, Y_test = train_test_split(problem_X_Train[l], y_Train[l], test_size=0.2, random_state=42)

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
    #linear_clf = LinearSVC(C=1, loss='hinge', random_state=42)
    #linear_clf.fit(X_train, Y_train)
    #train_predictions = linear_clf.predict(X_test)
    # train_predictions=svmPoly.predict(X_test)
    # clf = MLPClassifier(solver='lbfgs', alpha=1e-7, hidden_layer_sizes=(100, 50, 2), max_iter=3000, random_state=42)
    # model = BaggingClassifier(base_estimator=clf, n_estimators=500, random_state=1)

    model.fit(X_train, Y_train)
    train_predictions = model.predict(X_test)

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
