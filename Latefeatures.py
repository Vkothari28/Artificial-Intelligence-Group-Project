import pandas as pd
import SubjectProblemList
import numpy as np



def generating_num_attempts():
    df_main= pd.read_csv('S19_Release_6_28_21.zip/Train/Data/MainTable.csv')
    df = pd.read_csv('S19_Release_6_28_21.zip/Train/late.csv')
    j=0
    problem_list= SubjectProblemList.getProblems(df)
    subject_list= SubjectProblemList.getSubjects(df)

    print(problem_list)
    print(subject_list)
    print(len(subject_list))
    problem_list_index=0
    attempts=0;
    for i in range(len(subject_list)):
        for k in range(len(problem_list)):
            for j in range (len(df_main)):
                if df_main['ProblemID'].iloc[j] == problem_list[k] and df_main['SubjectID'].iloc[j]==subject_list[i]:
                    attempts+=1

            print('SubjectID',subject_list[i])
            print('ProblemID', problem_list[k])
            print('Attempts',attempts)
            attempts=0



def trying_something():
   array=np.empty(2)



if __name__=='__main__':
    ##generating_num_attempts()
    trying_something()

