import numpy as np
import pandas as pd
import pickle


def extractSkills():
    df = pd.read_csv('ConceptsUsed.csv')
    last_skills = []
    final_skills = []

    for i in range(30, len(df)):
        for column, row in df.iteritems():
            if df[column].iloc[i] == 1:
                last_skills.append(column)
        print(last_skills)
        final_skills.append(last_skills)
        last_skills = []
    print(final_skills)

    with open('LateSkills.pickle', 'wb') as handle:
        pickle.dump(final_skills, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    extractSkills()
