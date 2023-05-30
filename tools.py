import numpy as np
import pandas as pd
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def blinkfinder(data_matrix, minimum=9. / 90):
    """
    :param minimum: Minimum time interval that recognized as a real blink
    :param data_matrix: Put the input matrix here:
    :return: the time point intervals of blink events, the time  intervals of blink events (in seconds),
             blink numbers, blink frequency (time/minute)
    """

    blink_log = []
    for line in data_matrix:
        if line[5] == '-1':
            blink_log.append(line[0])
        # Right eye data used here

    if len(blink_log) > 1:
        blink_log.append(1e9)
        ans = []
        start = blink_log[0]
        for i in range(1, len(blink_log)):
            if int(blink_log[i]) != int(blink_log[i - 1]) + 1:
                ans.append([start, blink_log[i - 1]])
                start = blink_log[i]
        ans_time = []
        for interval in ans:
            time = (int(interval[1]) - int(interval[0]) + 1) / 90
            ans_time.append(time)
        time_processed = []
        for time_inter in ans_time:
            if float(time_inter) < minimum:
                continue
            else:
                time_processed.append(time_inter)
        return ans, time_processed, len(time_processed), len(time_processed) / ((len(data_matrix) - 1) / 5400)
    if len(blink_log) == 1:
        return [blink_log[0]], 1 / 90, 1, 1 / ((len(data_matrix) - 1) / 5400)
    else:
        return [], 0, 0, 0


def normalization(data):
    """

    :param data: Input data matrix
    :return: Normalized data matrix

    """

    out_data = []
    for row in data:
        row = np.array(row)
        row_up = row - np.min(row)
        row_down = np.max(row) - np.min(row)
        if row_down == 0:
            row_down = 1
        row_ = row_up / row_down
        row_ = row_.tolist()
        out_data.append(row_)
    return out_data


def anova_(data, name):
    """

    :param data: Input data matrix
    :param name: Name of the different conditions
    :return: Data matrix in [Name: data] format; Dta matrix in run-table format.
    """
    data_box = pd.DataFrame({name[0]: data[:, 0],
                             name[1]: data[:, 1],
                             name[2]: data[:, 2],
                             name[3]: data[:, 3],
                             name[4]: data[:, 4]})
    data_in = pd.DataFrame({'Subjects': np.tile(range(1, 25), 5),
                            'Conditions': np.repeat([0, 1, 2, 3, 4], 24),
                            'Frequency': np.reshape(data, 120, order='F')})
    print(AnovaRM(data=data_in, depvar='Frequency',
                  subject='Subjects', within=['Conditions']).fit())
    return data_box, data_in


def tukey_(data, print_result=True):
    """

    :param data: Input data matrix
    :param print_result: Whether or not to print the Tukey's result. Default: True
    :return: ~Reformatted Turky's result matrix for plotting; Turky's result matrix
    """
    tukey = pairwise_tukeyhsd(endog=data['Frequency'],
                              groups=data['Conditions'],
                              alpha=0.05)
    meandiffs = np.array(tukey.meandiffs).reshape((len(tukey.meandiffs)))
    pvalues = np.array(tukey.pvalues).reshape((len(tukey.meandiffs)))
    reject = np.array(tukey.reject).reshape((len(tukey.meandiffs)))
    interval = np.array(tukey.confint).T
    interval_min = np.array(interval[:][0]).reshape((len(tukey.meandiffs)))
    interval_max = np.array(interval[:][1]).reshape((len(tukey.meandiffs)))
    group1 = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3]).reshape((len(tukey.meandiffs)))
    group2 = np.array([1, 2, 3, 4, 2, 3, 4, 3, 4, 4]).reshape((len(tukey.meandiffs)))
    out_list = np.vstack((group1, group2, interval_min, interval_max, meandiffs, pvalues, reject), dtype=str).T
    if print_result is True:
        print("Tukey Result Start Here:")
        print(tukey)
    return out_list, [meandiffs, pvalues, reject, interval, group1, group2]
