import os
import random

import pandas as pd
import numpy as np
import sklearn.svm

from ProgSnap2 import ProgSnap2Dataset
from ProgSnap2 import PS2
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import pickle

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

early_train = pd.read_csv('../newEarlyTestF19.csv')

late_train = pd.read_csv('../../data/F19/Test/late.csv')

X_train_base = late_train.copy()


problem_encoder = OneHotEncoder().fit(X_train_base[PS2.ProblemID].values.reshape(-1, 1))
problem_encoder.transform(X_train_base[PS2.ProblemID].values.reshape(-1, 1)).toarray()


def extract_instance_features(instance, early_df):
    instance = instance.copy()
    subject_id = instance[PS2.SubjectID]
    early_problems = early_df[early_df[PS2.SubjectID] == subject_id]
    # Extract very naive features about the student
    # (without respect to the problem bring predicted)
    # Number of early problems attempted
    instance['ProblemsAttempted'] = early_problems.shape[0]
    # Percentage of early problems gotten correct eventually
    instance['PercCorrectEventually'] = np.mean(early_problems['CorrectEventually'])
    # Median attempts made on early problems
    instance['MedAttempts'] = np.median(early_problems['Attempts'])
    # Max attempts made on early problems
    instance['MaxAttempts'] = np.max(early_problems['Attempts'])
    # Percentage of problems gotten correct on the first try
    instance['PercCorrectFirstTry'] = np.mean(early_problems['Attempts'] == 1)
    df_generatedFeatures = pd.read_csv(os.path.join('../newEarlyTestF19.csv'))
    df_generatedFeatures_problems = df_generatedFeatures[df_generatedFeatures[PS2.SubjectID] == subject_id]
    # print(len(df_generatedFeatures_problems))
    instance['PercSubjectSyntaxErrors'] = np.median(df_generatedFeatures_problems['pSubjectSyntaxError'])
    instance['PercSubjectSemanticErrors'] = np.median(df_generatedFeatures_problems['pSubjectSemanticError'])
    instance['meanLabels'] = np.mean(early_problems['Label'])
    instance = instance.drop('SubjectID')
    return instance


def extract_features(X, early_df, scaler, is_train):
    # First extract performance features for each row
    features = X.apply(lambda instance: extract_instance_features(instance, early_df), axis=1)
    # Then one-hot encode the problem_id and append it
    # problem_ids = problem_encoder.transform(features[PS2.ProblemID].values.reshape(-1, 1)).toarray()
    problem_ids = features[PS2.ProblemID].values.reshape(-1, 1)
    # Then get rid of nominal features
    features.drop([PS2.AssignmentID, PS2.ProblemID], axis=1, inplace=True)
    # Then scale the continuous features, fitting the scaler if this is training
    if is_train:
        scaler.fit(features)
    features = scaler.transform(features)
    # Return continuous and one-hot features together
    return np.concatenate([features, problem_ids], axis=1)


early_test = pd.read_csv('../newEarlyTestF19.csv')
late_test = pd.read_csv( '../../data/F19/Test/late.csv')
X_test = extract_features(late_test, early_test, scaler, True)

print(X_test.shape)
print(X_test[:2, ])

with open('../Approach2/LateTestAllFeaturesF19.pickle', 'wb') as handle:
    pickle.dump(X_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
