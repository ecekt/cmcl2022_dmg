import numpy as np
import csv

answer = []
truth = []

with open('PAPER_RESULTS/results_mono_new/test_answers.csv', 'r') as f, \
        open('data/reference_data_subtask1_test2022/truth.txt') as r:

    lines = csv.reader(f, delimiter=',')
    lines_truth = csv.reader(r, delimiter=',')

    count = 0
    
    for line in lines:
        if count == 0:
            pass  # header
        else:
            answer.append(line)            

        count += 1

    count = 0
    
    for line in lines_truth:
        if count == 0:
            pass  # header
        else:
            truth.append(line)            

        count += 1
        
mae_ffdavg = []
mae_ffdstd = []
mae_trtavg = []
mae_trtstd = []

# 'language', 'sentence_id', 'word_id', 'word', 'FFDAvg', 'FFDStd', 'TRTAvg', 'TRTStd'

for l in range(len(answer)):
    
    ans = answer[l]
    tru = truth[l]

    ffdavg_dif = abs(float(ans[4]) - float(tru[4]))
    ffdstd_dif = abs(float(ans[5]) - float(tru[5]))
    trtavg_dif = abs(float(ans[6]) - float(tru[6]))
    trtstd_dif = abs(float(ans[7]) - float(tru[7]))

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

print(round(mean_mae_fa, 4), round(mean_mae_fs, 4), round(mean_mae_ta, 4), round(mean_mae_ts, 4), round(mean_mae, 4))

# official: 6.4774	4.7555	2.9235	1.9706	4.0317
# calculated:
# FFDAvg: 2.9235
# FFDStd: 1.9706
# TRTAvg: 6.4774
# TRTStd: 4.7555
# MAE: 4.0317
