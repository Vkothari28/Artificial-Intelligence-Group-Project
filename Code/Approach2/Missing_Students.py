import numpy as np
import pandas as pd
from SubjectProblemList import getSubjects, getProblems


def find_missing_students(Train):
    late_spring = pd.read_csv('../../data/S19/Train/late.csv')
    late_fall = pd.read_csv('../../data/F19/Test/late.csv')
    students_late_spring = getSubjects(late_spring)
    students_late_fall = getSubjects(late_fall)
    students_late_spring = students_late_fall
    problem_list = getProblems(late_spring)  # same list
    subject_index = 0

    if (Train == True):
        late_spring = late_fall
        students_late_spring = students_late_fall


    attempted_list_per_student_copy = [False for i in range(len(problem_list))]
    attempted_list_per_student = attempted_list_per_student_copy.copy()
    final_attempted_list = []
    for s in range(len(students_late_spring)):
        for i in range(len(late_spring)):
            if late_spring['SubjectID'].iloc[i] == students_late_spring[s] and late_spring['ProblemID'].iloc[i] in problem_list:
                attempted_list_per_student[problem_list.index(late_spring['ProblemID'].iloc[i])] = True

        final_attempted_list.append(attempted_list_per_student)
        attempted_list_per_student = attempted_list_per_student_copy.copy()

    return final_attempted_list


if __name__ == '__main__':
    find_missing_students(True)
