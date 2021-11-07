import os
import statistics
from _ast import mod

import pandas as pd


def getSubjects(df):
    subjectList = []
    for i in range(len(df)):
        if df['SubjectID'].iloc[i] not in subjectList:
            subjectList.append(df['SubjectID'].iloc[i])
    return subjectList


def getProblems(df):
    problemList = []
    for i in range(len(df)):
        if df['ProblemID'].iloc[i] not in problemList:
            problemList.append(df['ProblemID'].iloc[i])
    return problemList


def getPcorrect():
    df_to_edit= pd.read_csv('S19_Release_6_28_21.zip/Train/Data/MainTable.csv')
    df1= pd.read_csv('newEarlyTrain.csv')
    list_of_probs=getProblems(df1)
    df1["pCorrectProblem"]=""


    count=0
    correct_count=0
    p_correct_problem=0
    loc_list=[]
    i=0
    for i in range(len(list_of_probs)):
        for j in range(len(df1)):
            if(df1['ProblemID'].iloc[j]==list_of_probs[i]):
                count+=1
                loc_list.append(j)


                if(df1['Attempts'].iloc[j]==1 and df1['CorrectEventually'].iloc[j]==True):
                    correct_count+=1
        print("Count:"+str(count))
        print("Correct:"+str(correct_count))
        p_correct_problem=correct_count/count
        #print(p_correct_problem)

        checkcount=0
        for x in range(len(df1)):
            if(df1['ProblemID'].iloc[x]==list_of_probs[i]):
                df1['pCorrectProblem'].iloc[x]=p_correct_problem
                checkcount+=1
            if(checkcount>count):
                break

        p_correct_problem=0
        count = 0
        correct_count = 0
        checkcount=0
    pd.set_option('display.max_columns', 16)

    df1.to_csv('newEarlyTrain.csv')


    return df1


def pMedianAttemps(df):
    data=[]
    median_list=[]
    early_df= df
    prob_list=getProblems(early_df)
    count=0
    i=0
    early_df['pMedian']=""
    for i in range(len(prob_list)):
        for j in range(len(early_df)):
            if early_df['ProblemID'].iloc[j]==prob_list[i]:
                data.append(early_df['Attempts'].iloc[j])
                count+=1
        print(data)
        median_list.append(float(statistics.median(data)))
        data.clear()

        cc=0
        for x in range(len(early_df)):
            if cc>count:
                break
            elif early_df['ProblemID'].iloc[x]==prob_list[i]:
                early_df['pMedian'].iloc[x]=median_list[i]
                cc+=1
        count=0


        early_df.to_csv('newEarlyTrain.csv')
        print(early_df)



def syntax_sucks():
    df=pd.read_csv('newEarly.csv')
    #no of problems which a subject commited syntax error
    problist=getProblems(df)
    subject_list= getSubjects(df)
    df_main=pd.read_csv('S19_Release_6_28_21.zip/Train/Data/MainTable.csv')
    count=0
    probcheck=[]
    syntaxed=[]
    print(subject_list)
    print(len(subject_list))
    for x in range (len(subject_list)):
        for i in range(len(df_main)):
            if(df_main['SubjectID'].iloc[i]==subject_list[x] and df_main['ProblemID'].iloc[i] in problist  ):

                if(df_main['ProblemID'].iloc[i] not in probcheck):
                    probcheck.append(df_main['ProblemID'].iloc[i])
                    #count += 1
                if df_main['CompileMessageType'].iloc[i] == 'SyntaxError' and df_main['ProblemID'].iloc[i] not in syntaxed:
                    syntaxed.append(df_main['ProblemID'].iloc[i])

        print(len(syntaxed))
        print(len(probcheck))
        print(len(syntaxed)/len(probcheck))
        probcheck.clear()
        syntaxed.clear()
        count+=1
        print('Student no:', count)




def main2():

   ## getRandomly('newEarly.csv')
    df=getPcorrect()

    pMedianAttemps(df)
    #syntax_sucks()


def main():
    # df = pd.read_csv('./Data/MainTable.csv')
    df = pd.read_csv('newEarlyTrain.csv')
    subjectList = getSubjects(df)
    problemList = getProblems(df)
    # df2 = pd.DataFrame(subjectList, columns=["SubjectID"])
    df2 = pd.DataFrame({'SubjectID': subjectList})
    df2.to_csv('SubjectList.csv', index=False)
    df3 = pd.DataFrame(problemList, columns=["ProblemID"])
    df3.to_csv('ProblemList.csv', index=False)


if __name__ == "__main__":
   # main()
    main2()
