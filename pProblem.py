import pandas as pd

df = pd.read_csv('early.csv')

df.insert(4, 'pProblemSyntaxError', range(len(df)))
df.insert(5, 'pProblemSemanticError', range(len(df)))

df2 = pd.read_csv('MainTable.csv')



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


def main():
    print(p_prob_syntax_error())


if __name__ == "__main__":
    main()
