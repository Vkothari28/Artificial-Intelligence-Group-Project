import os
import pandas as pd
import numpy as np
from ProgSnap2 import ProgSnap2Dataset
from ProgSnap2 import PS2
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

semester = 'S19'
BASE_PATH = os.path.join('Data', 'Release', semester)
TRAIN_PATH = os.path.join(BASE_PATH, 'Train')
TEST_PATH = os.path.join(BASE_PATH, 'Test')

# The early dataset will help us to feature extraction,
# but we're not actually predicting anything here
# Note: we could still use this for model training if desired.
early_train = pd.read_csv(os.path.join(TRAIN_PATH, 'early.csv'))
# print(early_train.head())


# The late dataset contains the problems that we're actually predicting for.
# The training portion of it includes labels.
late_train = pd.read_csv(os.path.join(TRAIN_PATH, 'late.csv'))
late_train.head()


X_train_base = late_train.copy().drop('Label', axis=1)
y_train = late_train['Label'].values

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
    df_generatedFeatures = pd.read_csv(os.path.join(TRAIN_PATH, 'newEarlyTrain.csv'))
    df_generatedFeatures_problems = df_generatedFeatures[df_generatedFeatures[PS2.SubjectID] == subject_id]
    # print(len(df_generatedFeatures_problems))
    instance['PercSubjectSyntaxErrors'] = np.median(df_generatedFeatures_problems['pSubjectSyntaxError'])
    instance['PercProbSemanticErrors'] = np.median(df_generatedFeatures_problems['pSubjectSemanticError'])
    # instance['SubjectMeanAttempts'] = np.mean(df_generatedFeatures_problems['pMedian'])
    instance = instance.drop('SubjectID')
    return instance


print(extract_instance_features(X_train_base.iloc[1], early_train))
# print(extract_instance_features(X_train_base.iloc[2], early_train))


def extract_features(X, early_df, scaler, is_train):
    # First extract performance features for each row
    features = X.apply(lambda instance: extract_instance_features(instance, early_df), axis=1)
    # Then one-hot encode the problem_id and append it
    problem_ids = problem_encoder.transform(features[PS2.ProblemID].values.reshape(-1, 1)).toarray()
    # Then get rid of nominal features
    features.drop([PS2.AssignmentID, PS2.ProblemID], axis=1, inplace=True)
    # Then scale the continuous features, fitting the scaler if this is training
    if is_train:
        scaler.fit(features)
    features = scaler.transform(features)

    # Return continuous and one-hot features together
    return np.concatenate([features, problem_ids], axis=1)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = extract_features(X_train_base, early_train, scaler, True)


print(X_train.shape)
X_train[:2, ]

from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier

# model = LogisticRegressionCV()
# model = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
clf = MLPClassifier(solver='lbfgs', alpha=1e-2, hidden_layer_sizes=(50, 25, 2), random_state=42)
model = BaggingClassifier(base_estimator=clf, n_estimators=65, random_state=42)

model.fit(X_train, y_train)
train_predictions = model.predict(X_train)

from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

print(classification_report(y_train, train_predictions))
print('AUC: ' + str(roc_auc_score(y_train, train_predictions)))
print('Macro F1: ' + str(f1_score(y_train, train_predictions, average='macro')))

from sklearn.metrics import plot_roc_curve

plot_roc_curve(model, X_train, y_train)


from sklearn.model_selection import cross_validate

# model = LogisticRegressionCV()
# model = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
model = BaggingClassifier(base_estimator=clf, n_estimators=65, random_state=42)
cv_results = cross_validate(model, X_train, y_train, cv=10, scoring=['accuracy', 'f1_macro', 'roc_auc'])
print(f'Accuracy: {np.mean(cv_results["test_accuracy"])}')
print(f'AUC: {np.mean(cv_results["test_roc_auc"])}')
print(f'Macro F1: {np.mean(cv_results["test_f1_macro"])}')