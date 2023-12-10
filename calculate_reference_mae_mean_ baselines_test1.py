import numpy as np
import csv

# read train data to calculate mean baselines
# read test data to compare against mean baselines

selected_languages = ['en', 'de', 'nl', 'ru', 'hi', 'zh']

truth = []

ffdavgs = []
ffdstds = []
trtavgs = []
trtstds = []

with open('data/training_data2022/training_data/train.csv') as r:

    lines_train = csv.reader(r, delimiter=',')

    count = 0

    for line in lines_train:
        if count == 0:
            pass  # header
        else:
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

mae_ffdavg = []
mae_ffdstd = []
mae_trtavg = []
mae_trtstd = []

print(mean_ffdavg, mean_ffdstd, mean_trtavg, mean_trtstd)

# 'language', 'sentence_id', 'word_id', 'word', 'FFDAvg', 'FFDStd', 'TRTAvg', 'TRTStd'

test = []

with open('data/reference_data_subtask1_test2022/truth.txt') as r:

    lines_test = csv.reader(r, delimiter=',')

    count = 0

    for line in lines_test:
        if count == 0:
            pass  # header
        else:
            test.append(line)

        count += 1

for l in range(len(test)):

    tru = test[l]

    ffdavg_dif = abs(mean_ffdavg - float(tru[4]))
    ffdstd_dif = abs(mean_ffdstd - float(tru[5]))
    trtavg_dif = abs(mean_trtavg - float(tru[6]))
    trtstd_dif = abs(mean_trtstd - float(tru[7]))

    mae_ffdavg.append(ffdavg_dif)
    mae_ffdstd.append(ffdstd_dif)
    mae_trtavg.append(trtavg_dif)
    mae_trtstd.append(trtstd_dif)

mean_mae_fa = np.mean(mae_ffdavg)
mean_mae_fs = np.mean(mae_ffdstd)
mean_mae_ta = np.mean(mae_trtavg)
mean_mae_ts = np.mean(mae_trtstd)

mean_mae = np.mean([mean_mae_fa, mean_mae_fs, mean_mae_ta, mean_mae_ts])

print('FFDAvg:', round(mean_mae_fa, 4))
print('FFDStd:', round(mean_mae_fs, 4))
print('TRTAvg:', round(mean_mae_ta, 4))
print('TRTStd:', round(mean_mae_ts, 4))
print('MAE:', round(mean_mae, 4))
print()
# official: MAE_TRTAvg	MAE_TRTStd	MAE_FFDAvg	MAE_FFDStd
# Mean baseline	8.8200	5.8880	5.6860	2.5400	5.7330
# calculated:
# FFDAvg: 5.6858
# FFDStd: 2.5395
# TRTAvg: 8.82
# TRTStd: 5.8877
# MAE: 5.7332
print(round(mean_mae_fa, 4), round(mean_mae_fs, 4), round(mean_mae_ta, 4), round(mean_mae_ts, 4), round(mean_mae, 4))

