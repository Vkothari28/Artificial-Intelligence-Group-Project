import pandas as pd
import numpy as np
import pickle
from SubjectProblemList import getProblems, getSubjects
from Code.Approach2.Missing_Students import find_missing_students

missing = find_missing_students(True)
print('yfcvgb', len(missing))

late_train = pd.read_csv('../../data/S19/Train/late.csv')

late_test = pd.read_csv('../../data/F19/Test/late.csv')

with open('../Approach2/LateTrainAllFeatures.pickle', 'rb') as handle:
    X_Train = pickle.load(handle)

with open('../Approach2/LateTestAllFeaturesF19.pickle', 'rb') as handle:
    X_Test = pickle.load(handle)

TrainLateProblemList = pd.read_csv('../LateTrainProblemList.csv')

TrainLateSubjectList = getSubjects(late_train)
print(len(TrainLateSubjectList))
TestLateSubjectList = getSubjects(late_test)
print(len(TestLateSubjectList))


def perProblemFeatures(X_train, prob_id):
    # print('Problem ID: ', prob_id)
    n_previously_generated_features = 8
    subjectFeatures = []
    for i in range(len(X_train)):
        if X_train[i, n_previously_generated_features] == prob_id:
            subjectFeatures.append(X_train[i, :8])  # Try with different features from 5-8 (First 5 from Naive Model)
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


macrof1_model = []
macrof1_cv = []
AUC_model = []
AUC_cv = []
test_predictions = []
problem_list = getProblems(late_train)
last_predictions = []
for i in range(20):
    problem_predictions = []
    X_train, Y_train = problem_X_Train[i], y_Train[i]
    X_test = problem_X_Test[i]
    # if np.NaN in X_train: print('*****************************Hope')

    if i != 0:
        X_train = np.asarray(X_train)
        X_test = np.asarray(X_test)

        # X_test = np.append(X_test, [column[i] for column in missing], axis=1)
        X_train = np.append(X_train, [[0]] * len(X_train), axis=1)
        X_test = np.append(X_test, [[0]] * len(X_test), axis=1)

        ind_pred = 0
        n_previously_generated_features = 8
        # print(missing)
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

    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=500, max_leaf_nodes=64, n_jobs=-1)

    # print('X Train: ', X_train[0])
    # print('X Test: ', X_test[0])
    model.fit(X_train, Y_train)
    problem_predictions = model.predict(X_test)
    # print('Initial Prediction: ', problem_predictions)
    last_predictions = problem_predictions
    test_predictions.append(problem_predictions)
    # print('Predictions: ', (test_predictions))

# print(len(test_predictions[0]))

late_test['Label'] = ""

for i in range(len(TrainLateProblemList)):
    for j in range(len(test_predictions[i])):
        assigned = 0
        for l in range(len(late_train)):
            if TrainLateProblemList['ProblemID'].iloc[i] == late_train['ProblemID'].iloc[l]:
                late_train['Label'].iloc[l] = bool(test_predictions[i][j])
                assigned += 1
            if assigned > len(test_predictions[i]):
                break

late_train.to_csv('../Approach2/PredictedF19.csv', index=None)
