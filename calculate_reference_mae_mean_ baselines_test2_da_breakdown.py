import numpy as np
import csv

# read train data to calculate mean baselines
# read test data to compare against mean baselines

selected_languages = ['da']

corpora = ['CopCo']

for corpus in corpora:
    print(corpus)
    for selected_lang in selected_languages:

        print(selected_lang)

        truth = []

        ffdavgs = []
        ffdstds = []
        trtavgs = []
        trtstds = []

        with open('data/reference_data_subtask2_test2022/truth.txt') as r:

            lines_train = csv.reader(r, delimiter=',')

            count = 0

            for line in lines_train:
                if count == 0:
                    pass  # header
                else:
                    if line[0] == selected_lang and corpus in line[1]:
                        truth.append(line)

                        ffdavgs.append(float(line[4]))
                        ffdstds.append(float(line[5]))
                        trtavgs.append(float(line[6]))
                        trtstds.append(float(line[7]))

                count += 1

        print(len(truth))

        mean_ffdavg = np.mean(ffdavgs)
        mean_ffdstd = np.mean(ffdstds)
        mean_trtavg = np.mean(trtavgs)
        mean_trtstd = np.mean(trtstds)

        print(mean_ffdavg, mean_ffdstd, mean_trtavg, mean_trtstd, '\n')
