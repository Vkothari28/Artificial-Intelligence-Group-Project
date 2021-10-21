import pandas as pd


df= pd.read_csv('addCSVNAME')
def assignment():
    list=[]


    list2=[ 'If /Else',
    'NestedIf',
    'While',
    'For',
    'NestedFor',
    'Math+-*/',
    'Math%',
    'LogicAndNotOr'
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
    for (columnName,columnData) in df.iteritems():


        if columnName in list2: ## all the names:

            for i in range(len(df[columnName])):
                if df[columnName].iloc[i] :
                    list.append(df[columnName].iloc[i])




    print(list)



if __name__=='__main__':
    assignment()
