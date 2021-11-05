import pandas as pd
import numpy as np
import pandas
import csv
import ast
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, cohen_kappa_score, accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
import pickle

input_feat = [0, 1, 2, 3]
output_feat = [4]
no_estimators = 65


def basic_model(df):
    train_x = df.iloc[:, input_feat]
    train_y = df.iloc[:, output_feat]
    train_x = train_x.values
    train_y = train_y.values
    # clf = MLPClassifier(solver='lbfgs' , alpha=1e-5,hidden_layer_sizes=(100,50,2), random_state=1).fit(train_x,train_y)
    # clf=svm.SVC(kernel='rbf').fit(train_x,train_y)
    clf = DecisionTreeClassifier().fit(train_x, train_y)
    # clf=LogisticRegression(solver='lbfgs')
    model = BaggingClassifier(base_estimator=clf, n_estimators=no_estimators, random_state=7)
    # model = AdaBoostClassifier(base_estimator=clf, n_estimators=no_estimators, learning_rate=5)
    model = model.fit(train_x, train_y)
    # model = clf.fit(train_x, train_y)
    return model


# accuracy_list = []
# f1_score_list = []
# precision_score_list = []
# kappa_score_list = []
# recall_score_list = []
# tp = []
# fp = []
# fn = []
# tn = []
# frames = []


df_train = pd.read_csv("newEarlyTrain.csv")
df_test = pd.read_csv("newEarlyTest.csv")

df_train = df_train[["Attempts", "pProblemSyntaxError", "pProblemSemanticError", "CorrectEventually", "Label"]]

df_test = df_test[["Attempts", "pProblemSyntaxError", "pProblemSemanticError", "CorrectEventually", "Label"]]

model = basic_model(df_train)

test_x = df_test.iloc[:, input_feat]
test_x = test_x.values
test_y = df_test.iloc[:, output_feat]
test_y = test_y.values

prediction = model.predict(test_x)
prediction = list(prediction)

# print(len(df_test))
count = 0
for i in range(len((df_test))):
    if df_test['Label'].iloc[i] is not prediction[i]:
        count += 1
print(count)
# print(prediction[8])
# pickle.dump(model, open(("model" + str(i) + ".pkl"), "wb"))
#
# test_x=df_test.iloc[:,input_feat]
# test_x=test_x.values
# test_y = df_test.iloc[:,output_feat]
# test_y =test_y.values
#
# prediction =  model.predict(test_x)
# df_test["prediction"] = prediction
# fold_id = [i]*len(prediction)
# df_test["Fold"] =fold_id
#
# prediction = list(prediction)
# test_synt_list=[False]*df_test_syntax_error.shape[0]
#
# df_test_syntax_error["prediction"] = test_synt_list
# fold_id = [i]*len(test_synt_list)
# df_test_syntax_error["Fold"] =fold_id
#
# prediction+=test_synt_list
#
# test_y = [i[0] for i in test_y]
#
# test_y+=list(df_test_syntax_error['Correct'])
#
# accuracy= accuracy_score(test_y, prediction)
# accuracy_list.append(accuracy)
#
# f1=f1_score(test_y,prediction)
# f1_score_list.append(f1)
#
# precision = precision_score(test_y,prediction)
# precision_score_list.append(precision)
#
# kappa=cohen_kappa_score(test_y,prediction)
# kappa_score_list.append(kappa)
#
# recall = recall_score(test_y,prediction)
# recall_score_list.append(recall)
#
# cm=confusion_matrix(test_y, prediction)
# tp.append(cm[0][0])
# fp.append(cm[0][1])
# fn.append(cm[1][0])
# tn.append(cm[1][1])
#
# result=pd.concat(frames)
# result.to_csv("cv_predict.csv",index = False)
#
# d={"accuracy" :[mean(accuracy_list)]  , "f1_score" :mean(f1_score_list) , "precision_score" : mean(precision_score_list)  , "kappa_score" : mean(kappa_score_list) ,"recall_score" :mean(recall_score_list) }
# df=pd.DataFrame(data=d)
#
# df.to_csv("evaluation_overall.csv",index = False)
#
#
# df=pd.read_csv("cv_predict.csv")
# df=df[["ProblemID","Correct", "prediction"]]
#
# col=[]
# tn=[]
# tp=[]
# fn=[]
# fp=[]
#
# for index,rows in df.iterrows():
#     if(rows["Correct"] == True  and rows["prediction"]==True):
#         tp.append(1)
#     else:
#         tp.append(0)
#     if(rows["Correct"] == False  and rows["prediction"]==True):
#         fp.append(1)
#     else:
#         fp.append(0)
#     if(rows["Correct"] == False and rows["prediction"]== False):
#         tn.append(1)
#     else:
#         tn.append(0)
#     if(rows["Correct"] == True  and rows["prediction"]== False):
#         fn.append(1)
#     else:
#         fn.append(0)
#
#     if(rows["Correct"] == rows["prediction"]):
#         col.append(1)
#     else:
#         col.append(0)
#
#
# df["accuracy"]=col
# df["tp"] = tp
# df["tn"] = tn
# df["fp"] = fp
# df["fn"] = fn
#
#
# df_accuracy=df.groupby("ProblemID",as_index=False)["accuracy"].mean()
# # df=pd.merge(df,df_accuracy,on=["ProblemID", "" ])
# df_tn  =df.groupby("ProblemID",as_index=False)["tn"].mean()
# df_tp  =df.groupby("ProblemID",as_index=False)["tp"].mean()
# df_fn  =df.groupby("ProblemID",as_index=False)["fn"].mean()
# df_fp  =df.groupby("ProblemID",as_index=False)["fp"].mean()
# df_pcorrect=df.groupby("ProblemID",as_index=False)["Correct"].mean()
# df_ppredicted = df.groupby("ProblemID" ,as_index=False)["prediction"].mean()
#
#
# accuracy=list(df_accuracy["accuracy"])
# tp=list(df_tp["tp"])
# tn=list(df_tn["tn"])
# fp=list(df_fp["fp"])
# fn=list(df_fn["fn"])
# pcorrect=list(df_pcorrect["Correct"])
# ppredicted=list(df_ppredicted["prediction"])
#
#
# df_accuracy["tp"] = tp
# df_accuracy["tn"] = tn
# df_accuracy["fp"] = fp
# df_accuracy["fn"] = fn
# df_accuracy["pcorrect"] = pcorrect
# df_accuracy["ppredicted"] = ppredicted
#
#
# df_test =df_accuracy
# df_accuracy=df_accuracy.assign( precision = ( df_accuracy["tp"] )/ (df_accuracy["tp"] + df_accuracy["fp"]))
# df_accuracy=df_accuracy.assign( recall  = ( df_accuracy["tp"] )/ (df_accuracy["tp"] + df_accuracy["fn"]))
#
# df_accuracy=df_accuracy.assign( f1_score = ( 2*df_accuracy["precision"] *df_accuracy["recall"] )/ (df_accuracy["precision"] + df_accuracy["recall"]))
# df_accuracy=df_accuracy.assign( pe = ( df_accuracy["pcorrect"] * df_accuracy["ppredicted"] ) + (1-df_accuracy["pcorrect"])*(1-df_accuracy["ppredicted"] ))
# df_accuracy=df_accuracy.assign( kappa = ( df_accuracy["accuracy"] - df_accuracy["pe"])/ (1- df_accuracy["pe"]))
#
#
# df_accuracy.to_csv("evaluation_by_problem.csv")
