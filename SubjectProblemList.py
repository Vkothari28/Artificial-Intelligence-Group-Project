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


def main():
    # df = pd.read_csv('./Data/MainTable.csv')
    df = pd.read_csv('early.csv')
    subjectList = getSubjects(df)
    problemList = getProblems(df)
    # df2 = pd.DataFrame(subjectList, columns=["SubjectID"])
    df2 = pd.DataFrame({'SubjectID': subjectList})
    df2.to_csv('SubjectList.csv', index=False)
    df3 = pd.DataFrame(problemList, columns=["ProblemID"])
    df3.to_csv('ProblemList.csv', index=False)


if __name__ == "__main__":
    main()
