import pandas as pd


df2= pd.read_csv('SkillList.csv')
def assignment(problemID):
    list=[]


    list2=[ 'If/Else',
    'NestedIf',
    'While',
    'For',
    'NestedFor',
    'Math+-*/',
    'Math%',
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


    index=0
    for i in range(len(df2['ProblemID'])):
        print(df2['ProblemID'].iloc[i])
        if(df2['ProblemID'].iloc[i]==problemID):
            for j in range (len(list2)):
                word=list2[j]

                if(df2[word].iloc[i]==1):
                    if(df2[word].iloc[i] not in list):

                        list.insert(index,word)
                        index+=1








    print(list)



if __name__=='__main__':
    assignment(12)
