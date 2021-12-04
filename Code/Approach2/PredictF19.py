import pandas as pd
import numpy as np
import pickle

late_train = pd.read_csv('../../data/S19/Train/late.csv')

with open('../Approach1/LateTrainAllFeatures.pickle', 'rb') as handle:
    X_Train = pickle.load(handle)

# print('X_train**************', X_Train[0])

with open('LateTestAllFeaturesF19.pickle', 'rb') as handle:
    X_Test = pickle.load(handle)

# print(len(X_Train))
TrainLateProblemList = pd.read_csv('../LateTrainProblemList.csv')


def perProblemFeatures(X_train, prob_id):
    # print('Problem ID: ', prob_id)
    n_previously_generated_features = 6
    subjectFeatures = []
    for i in range(len(X_train)):
        if X_train[i, n_previously_generated_features] == prob_id:
            subjectFeatures.append(X_train[i, :6])  # Try with different features from 5-8 (First 5 from Naive Model)
    return subjectFeatures


problem_X_Train = []
problem_X_Test = []
sum = 0

for i in range(20):
    # print(TrainLateProblemList['ProblemID'].iloc[i])
    problem_X_Train.append(perProblemFeatures(X_Train, TrainLateProblemList['ProblemID'].iloc[i]))
    sum += len(problem_X_Train[i])
    # print(len(problem_X_Train[i]))
    problem_X_Test.append(perProblemFeatures(X_Test, TrainLateProblemList['ProblemID'].iloc[i]))
    sum += len(problem_X_Test[i])

y_Train = []

for i in range(len(TrainLateProblemList)):
    y_TrainPerProblem = []
    for j in range(len(late_train)):
        if late_train['ProblemID'].iloc[j] == TrainLateProblemList['ProblemID'].iloc[i]:
            y_TrainPerProblem.append(late_train['Label'].iloc[j])
    y_Train.append(y_TrainPerProblem)

# print('X_Train:', len(problem_X_Train[0]))
# print(sum)
# print(len(y_Train))
# print(len(y_Train[3]))


from sklearn.model_selection import train_test_split

macrof1_model = []
macrof1_cv = []
AUC_model = []
AUC_cv = []
test_predictions = []

for l in range(20):
    X_train, Y_train = problem_X_Train[l], y_Train[l]
    X_test = problem_X_Test[l]

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
    # print('X Train: ', X_train[0])
    # print('Y_train: ', Y_train[0])
    # print('X Test: ', X_test[0])
    model.fit(X_train, Y_train)
    problem_predictions = model.predict(X_test)
    test_predictions.append(problem_predictions)
    # print('Predictions: ', (test_predictions))

# print(len(test_predictions[0]))

late_test = pd.read_csv('../../data/F19/Test/late.csv')
late_test['Label'] = ""

for i in range(len(TrainLateProblemList)):
    for j in range(len(test_predictions[i])):
        assigned = 0
        for l in range(len(late_train)):
            if TrainLateProblemList['ProblemID'].iloc[i] == late_train['ProblemID'].iloc[l]:
                # print(test_predictions[i][j])
                late_train['Label'].iloc[l] = bool(test_predictions[i][j])
                assigned += 1
            if assigned > len(test_predictions[i]):
                break

late_train.to_csv('PredictedF19.csv', index=None)
