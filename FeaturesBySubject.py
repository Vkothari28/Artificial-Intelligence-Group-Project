import pandas as pd


# def generateSyntaxErrors(subjectList, problemList, df):
#     pSyntaxErrors = []
#     pSemanticErrors = []
#     countSyntaxErrors = [0] * len(subjectList)
#     countProblemsAttempted = [0] * len(problemList)
#     # print((countSyntaxErrors))
#     for subjectID in subjectList:
#         for problemID in problemList:
#             for column, rows in df.iterrows():
#                 # print(column, rows['SubjectID'])
#                 if rows['SubjectID'] == subjectID and rows['ProblemID'] == problemID:
#                     # print(subjectList.index(rows['SubjectID']))
#                     if rows['CompileMessageType'] == 'SyntaxError':
#                         countSyntaxErrors[subjectList.index(rows['SubjectID'])] += 1
#                         # print(problemID, column)
#                         break


def generateSyntaxFeaturesBySubject(subjectList, problemList, df):
    df2 = pd.read_csv('newEarly.csv')
    df2['pSubjectSyntaxErrors'] = ""

    count = 0
    probCheck = []
    syntaxed = []

    # print(subjectList)
    # print(len(subjectList))

    for x in range(len(subjectList)):
        for i in range(len(df)):
            if df['SubjectID'].iloc[i] == subjectList[x] and df['ProblemID'].iloc[i] in problemList:
                count += 1
                if df['ProblemID'].iloc[i] not in probCheck:
                    probCheck.append(df['ProblemID'].iloc[i])
                    # count += 1
                if df['CompileMessageType'].iloc[i] == 'SyntaxError' and df['ProblemID'].iloc[
                    i] not in syntaxed:
                    syntaxed.append(df['ProblemID'].iloc[i])

        # print(len(syntaxed))
        # print(len(probCheck))
        # print(len(syntaxed) / len(probCheck))

        checkCount = 0
        for idx in range(len(df2)):
            if df2['SubjectID'].iloc[idx] == subjectList[x]:
                df2['pSubjectSyntaxErrors'].iloc[idx] = len(syntaxed) / len(probCheck)
                checkCount += 1

            if checkCount > count:
                break

        count = 0
        probCheck.clear()
        syntaxed.clear()

    print(df2)
    df2.to_csv('newEarly.csv', index=None)


def generateSemanticFeaturesBySubject(subjectList, problemList, df):
    df2 = pd.read_csv('early.csv')
    df2['pSubjectSemanticErrors'] = ""

    count = 0
    probCheck = []
    semantic = []
    pSubjectSemanticErrors = []
    # print(subjectList)
    # print(len(subjectList))
    pSubjectSyntaxErrors = []
    for x in range(len(subjectList)):
        for i in range(len(df)):
            if df['SubjectID'].iloc[i] == subjectList[x] and df['ProblemID'].iloc[i] in problemList:
                count += 1
                if df['ProblemID'].iloc[i] not in probCheck:
                    probCheck.append(df['ProblemID'].iloc[i])

                if 0 < df['Score'].iloc[i] < 1 and df['ProblemID'].iloc[i] not in semantic:
                    semantic.append(df['ProblemID'].iloc[i])

        # print(len(semantic))
        # print(len(probCheck))
        # print(len(semantic) / len(probCheck))

        checkCount = 0
        for idx in range(len(df2)):
            if df2['SubjectID'].iloc[idx] == subjectList[x]:
                df2['pSubjectSemanticErrors'].iloc[idx] = len(semantic) / len(probCheck)
                checkCount += 1

            if checkCount > count:
                break

        count = 0
        probCheck.clear()
        semantic.clear()

    print(df2)
    df2.to_csv('newEarly.csv', index=None)


def main():
    df = pd.read_csv('./Data/MainTable.csv')
    df2 = pd.read_csv('SubjectList.csv')
    df3 = pd.read_csv('ProblemList.csv')
    df4 = pd.read_csv('early.csv')
    subjectList = df2['SubjectID'].to_list()
    problemList = df3['ProblemID'].to_list()
    # print(len(subjectList))
    # print(len(problemList))
    # generateSemanticFeaturesBySubject(subjectList, problemList, df)
    generateSyntaxFeaturesBySubject(subjectList, problemList, df)


if __name__ == "__main__":
    main()
