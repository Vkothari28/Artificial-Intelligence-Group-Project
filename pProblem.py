import pandas as pd

df = pd.read_csv('S19_Release_6_28_21.zip/Train/early.csv')

df.insert(4, 'pProblemSyntaxError', range(len(df)))
df.insert(5, 'pProblemSemanticError', range(len(df)))

df2 = pd.read_csv('S19_Release_6_28_21.zip/Train/Data/MainTable.csv')



def get_problems(df):
    prob_list = []
    for i in range(len(df)):
        if df['ProblemID'].iloc[i] not in prob_list:
            prob_list.append(df['ProblemID'].iloc[i])
    return prob_list


def getSubjects(df):
    subjectList = []
    for i in range(len(df)):
        if df['SubjectID'].iloc[i] not in subjectList:
            subjectList.append(df['SubjectID'].iloc[i])
    return subjectList


def p_prob_syntax_error_shortened():
    problist=get_problems(df)
    syntax_list=[]
    sublist=list()
    index=0

    for x in range(len(problist)):

        for a in range(len(df2)):
            if df2['ProblemID'].iloc[a]==problist[x]:
                if df2['SubjectID'].iloc[a] not in sublist:
                    sublist.append((df2['SubjectID'].iloc[a]))
                if(df2['CompileMessageType'].iloc[a]=='SyntaxError'):

                    if df2['SubjectID'].iloc[a] not in syntax_list:
                        syntax_list.append(df2['SubjectID'].iloc[a])
                        index+=1

        print('Problem ID'+ str(problist[x]))
        print('No. of students who answered', len(sublist))
        print('No of students who made syntax error', len(syntax_list))
        print('Percentage of synatx error by subjects', len(syntax_list)/len(sublist))
        sublist.clear()
        syntax_list.clear()




''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def p_prob_syntax_error():
    subjectID = getSubjects(df)
    problemList = get_problems(df)
    number_students = 246
    pSyntax = 0
    dic = {}
    # l = []

    for problemID in problemList:
       for subject in subjectID:
            for column, rows in df2.iterrows():
                if rows['SubjectID'] == subject and rows['ProblemID'] == problemID:
                    if rows['CompileMessageType'] == 'SyntaxError':
                        pSyntax += 1
                        break
        dic[problemID] = pSyntax/number_students
        # l.append(pSyntax/number_students)
    # df3 = pd.DataFrame(l,columns=problemList)
    # df3.to_csv('pProblemSyntaxError.csv', index=False)
    return dic
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def main():
    #print(p_prob_syntax_error())
    p_prob_syntax_error_shortened()

if __name__ == "__main__":
    main()
