import pandas as pd
import math

df_train = pd.read_csv('S19_Release_6_28_21.zip/Train/early.csv')
df_test = pd.read_csv('F19/Test/early.csv')

# df_test.insert(4, 'pProblemSyntaxError', range(len(df_test)))
# df_test.insert(5, 'pProblemSemanticError', range(len(df_test)))
# df_test.insert(6, 'pSubjectSyntaxError', range(len(df_test)))
# df_test.insert(7, 'pSubjectSemanticError', range(len(df_test)))

df_train.insert(4, 'pProblemSyntaxError', range(len(df_train)))
df_train.insert(5, 'pProblemSemanticError', range(len(df_train)))
df_train.insert(6, 'pSubjectSyntaxError', range(len(df_train)))
df_train.insert(7, 'pSubjectSemanticError', range(len(df_train)))

df_test.insert(4, 'pProblemSyntaxError', range(len(df_test)))
df_test.insert(5, 'pProblemSemanticError', range(len(df_test)))
df_test.insert(6, 'pSubjectSyntaxError', range(len(df_test)))
df_test.insert(7, 'pSubjectSemanticError', range(len(df_test)))




df2_test = pd.read_csv('F19/Test/Data/MainTable.csv')
df2_train = pd.read_csv('S19_Release_6_28_21.zip/Train/Data/MainTable.csv')


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


pd.options.mode.chained_assignment = None  # default='warn'


def p_prob_syntax_error_shortened(df, df2):
    problem_list = get_problems(df)
    syntax_list = []
    sublist = list()
    index = 0

    for x in range(len(problem_list)):

        for a in range(len(df2)):
            if df2['ProblemID'].iloc[a] == problem_list[x]:
                if df2['SubjectID'].iloc[a] not in sublist:
                    sublist.append((df2['SubjectID'].iloc[a]))
                if df2['CompileMessageType'].iloc[a] == 'SyntaxError':

                    if df2['SubjectID'].iloc[a] not in syntax_list:
                        syntax_list.append(df2['SubjectID'].iloc[a])
                        index += 1

        for i in range(len(df)):
            if df['ProblemID'].iloc[i] == problem_list[x]:
                df['pProblemSyntaxError'].iloc[i] = '{0:.3g}'.format(len(syntax_list)/len(sublist))
                df.to_csv('newEarlyTestF19.csv')

        # print('Problem ID' + str(problem_list[x]))
        # print('No. of students who answered', len(sublist))
        # print('No of students who made syntax error', len(syntax_list))
        # print('index is ', str(index))
        # print('Percentage of syntax error by subjects', len(syntax_list)/len(sublist))
        sublist.clear()
        syntax_list.clear()


def p_prob_semantic_error_shortened(df, df2):
    problem_list = get_problems(df)
    semantic_list = []
    sublist = list()
    index = 0

    for x in range(len(problem_list)):

        for a in range(len(df2)):
            if df2['ProblemID'].iloc[a] == problem_list[x]:
                if df2['SubjectID'].iloc[a] not in sublist:
                    sublist.append((df2['SubjectID'].iloc[a]))
                if 1 > df2['Score'].iloc[a] > 0:

                    if df2['SubjectID'].iloc[a] not in semantic_list:
                        semantic_list.append(df2['SubjectID'].iloc[a])
                        index += 1

        for i in range(len(df)):
            if df['ProblemID'].iloc[i] == problem_list[x]:
                df['pProblemSemanticError'].iloc[i] = '{0:.3g}'.format(len(semantic_list)/len(sublist))
                df.to_csv('newEarlyTestF19.csv')

        # print('Problem ID' + str(problem_list[x]))
        # print('No. of students who answered', len(sublist))
        # print('No of students who made semantic error', len(semantic_list))
        # print('index is ', str(index))
        # print('Percentage of syntax error by subjects', len(semantic_list)/len(sublist))
        sublist.clear()
        semantic_list.clear()
# def p_prob_syntax_error():
#     subjectID = getSubjects(df)
#     problemList = get_problems(df)
#     number_students = 246
#     pSyntax = 0
#     dic = {}
#     # l = []
#
#     for problemID in problemList:
#         for subject in subjectID:
#             for column, rows in df2.iterrows():
#                 if rows['SubjectID'] == subject and rows['ProblemID'] == problemID:
#                     if rows['CompileMessageType'] == 'SyntaxError':
#                         pSyntax += 1
#                         break
#         dic[problemID] = pSyntax/number_students
#         # l.append(pSyntax/number_students)
#     # df3 = pd.DataFrame(l,columns=problemList)
#     # df3.to_csv('pProblemSyntaxError.csv', index=False)
#     return dic


def generateSyntaxFeaturesBySubject(df, df2):
    problemList = get_problems(df)
    subjectList = getSubjects(df)
    count = 0
    probCheck = []
    syntaxed = []

    # print(subjectList)
    # print(len(subjectList))
    # for x in range(len(problem_list)):
    #
    #     for a in range(len(df2)):
    #         if df2['ProblemID'].iloc[a] == problem_list[x]:
    #             if df2['SubjectID'].iloc[a] not in sublist:
    #                 sublist.append((df2['SubjectID'].iloc[a]))

    for x in range(len(subjectList)):
        for i in range(len(df2)):
            if df2['SubjectID'].iloc[i] == subjectList[x] and df2['ProblemID'].iloc[i] in problemList:
                count += 1
                if df2['ProblemID'].iloc[i] not in probCheck:
                    probCheck.append(df2['ProblemID'].iloc[i])
                    # count += 1
                if df2['CompileMessageType'].iloc[i] == 'SyntaxError' and df2['ProblemID'].iloc[
                    i] not in syntaxed:
                    syntaxed.append(df2['ProblemID'].iloc[i])


        for idx in range(len(df)):
            if df['SubjectID'].iloc[idx] == subjectList[x]:
                df['pSubjectSyntaxError'].iloc[idx] = '{0:.3g}'.format(len(syntaxed)/len(probCheck))
                df.to_csv('newEarlyTestF19.csv')

        probCheck.clear()
        syntaxed.clear()


def generateSemanticFeaturesBySubject(df, df2):
    problemList = get_problems(df)
    subjectList = getSubjects(df)

    count = 0
    probCheck = []
    semantic = []
    pSubjectSemanticErrors = []
    # print(subjectList)
    # print(len(subjectList))
    for x in range(len(subjectList)):
        for i in range(len(df2)):
            if df2['SubjectID'].iloc[i] == subjectList[x] and df2['ProblemID'].iloc[i] in problemList:
                count += 1
                if df2['ProblemID'].iloc[i] not in probCheck:
                    probCheck.append(df2['ProblemID'].iloc[i])

                if 0 < df2['Score'].iloc[i] < 1 and df2['ProblemID'].iloc[i] not in semantic:
                    semantic.append(df2['ProblemID'].iloc[i])

        # print(len(semantic))
        # print(len(probCheck))
        # print(len(semantic) / len(probCheck))

        checkCount = 0
        for idx in range(len(df)):
            if df['SubjectID'].iloc[idx] == subjectList[x]:
                df['pSubjectSemanticError'].iloc[idx] = '{0:.3g}'.format(len(semantic)/len(probCheck))
                checkCount += 1

            if checkCount > count:
                break

        count = 0
        probCheck.clear()
        semantic.clear()

    df.to_csv('newEarlyTestF19.csv')

def main():
    # p_prob_syntax_error_shortened(df_train, df2_train)
    # p_prob_semantic_error_shortened(df_train, df2_train)
    generateSyntaxFeaturesBySubject(df_test, df2_test)
    generateSemanticFeaturesBySubject(df_test, df2_test)


if __name__ == "__main__":
    main()
