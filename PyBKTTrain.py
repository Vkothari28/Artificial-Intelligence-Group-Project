import pandas as pd
import numpy as np
from ProgSnap2 import ProgSnap2Dataset
from ProgSnap2 import PS2
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os
from os import path

from pyBKT.models import Model

model = Model(seed=42, num_fits=1)

df = pd.read_csv('early.csv')

df.insert(0, 'row', range(len(df)))
df2 = pd.read_csv('SkillList.csv')

df2 = df2.rename({'Math+-*/': 'Math1', 'Math%': 'Math2'}, axis=1)  # Renaming the columns to remove the error

list1 = []

list2 = ['If/Else',
         'NestedIf',
         'While',
         'For',
         'NestedFor',
         'Math1',
         'Math2',
         'LogicAndNotOr',
         'LogicCompareNum',
         'LogicBoolean',
         'StringFormat',
         'StringConcat',
         'StringIndex',
         'StringLen',
         'StringEqual',
         'CharEqual',
         'ArrayIndex',
         'DefFunction']

list3 = df.loc[:, 'ProblemID']
# print(df.loc[:, 'ProblemID'])

problemToSkill = {}

for i in range(len(df2['ProblemID'])):
    for j in range(len(list2)):
        word = list2[j]
        if df2[word].iloc[i] == 1:
            if df2[word].iloc[i] not in list1:
                list1.append(word)
    problemToSkill[df2['ProblemID'].iloc[i]] = list1
    list1 = []

# print(problemToSkill[1])
for problemID in df.iloc[:, 3]: # ww
    # print(problemID)
    # if problemID in list3:
    if problemID in problemToSkill.keys():
        list1.append(problemToSkill[problemID])
    # break

# print(len(list1))

df['SkillName'] = list1

# print(df['SkillName'])
# print(list1)
# df.to_csv(path_or_buf='NewEarly.csv')

defaults = {'order_id': 'row', 'user_id': 'SubjectID', 'problem_id': 'ProblemID', 'correct': 'CorrectEventually',
            'skill_name': 'SkillName'}


# print(df.head())
model.fit(data=df, defaults=defaults)
dd = model.params()

dd.to_csv(path_or_buf="modelparams.csv")

# training_acc = model.evaluate(data=df, metric='accuracy')

preds_df = model.predict(data=df)

preds_df.to_csv(path_or_buf="Results.csv")

# print(preds_df.head(20))
