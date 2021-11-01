import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('early.csv')
# dd = df.drop(['Label'], axis='columns')
target = df['Label']
# print(df.head())

X_train, X_test, y_train, y_test = train_test_split(df.drop(['Label', 'SubjectID'], axis='columns'), target, test_size=0.2)
# print(len(X_train))
# print(len(X_test))
# print(df.columns)

# model = RandomForestClassifier(n_estimators=20) # Using 20 trees
model = RandomForestClassifier()
model.fit(X_train, y_train)

print(model.score(X_test, y_test))

print(model.predict([[439.0, 1, 3, False]]))

# 439.0,1,1,True,True
