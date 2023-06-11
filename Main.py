import csv
import matplotlib.pyplot as plt
import pandas as pd

import statistic_tools
import numpy as np
import scipy
import seaborn as sns


root_dir = 'EyeTracking-data/P'
interval = []
frequency = []
for subject_num in range(1, 25):
    # Definite the directory of all csv files
    control_dir = root_dir+str(subject_num)+'/Eyerecording_Test_Control.csv'
    npc_dir = root_dir+str(subject_num)+'/Eyerecording_Test_NPC.csv'
    noise_dir = root_dir+str(subject_num)+'/Eyerecording_Test_Noise.csv'
    secondTask_dir = root_dir+str(subject_num)+'/Eyerecording_Test_Second_task.csv'
    combined_dir = root_dir+str(subject_num)+'/Eyerecording_Test_4_Combined.csv'

    # Pre-define value matrices
    control = []
    npc = []
    noise = []
    secondTask = []
    combined = []

    # Put the values into the matrices, with the head line
    control_reader = csv.reader(open(control_dir), delimiter=';')
    for line in control_reader:
        control.append(line[:-1])
    npc_reader = csv.reader(open(npc_dir), delimiter=';')
    for line in npc_reader:
        npc.append(line[:-1])
    noise_reader = csv.reader(open(noise_dir), delimiter=';')
    for line in noise_reader:
        noise.append(line[:-1])
    secondTask_reader = csv.reader(open(secondTask_dir), delimiter=';')
    for line in secondTask_reader:
        secondTask.append(line[:-1])
    combined_reader = csv.reader(open(combined_dir), delimiter=';')
    for line in combined_reader:
        combined.append(line[:-1])

    # Find the blink and calculate the blink related data
    log_control, time_control, num_control, f_control = statistic_tools.blinkfinder(control)
    log_npc, time_npc, num_npc, f_npc = statistic_tools.blinkfinder(npc)
    log_noise, time_noise, num_noise, f_noise = statistic_tools.blinkfinder(noise)
    log_secondTask, time_secondTask, num_secondTask, f_secondTask = statistic_tools.blinkfinder(secondTask)
    log_combined, time_combined, num_combined, f_combined = statistic_tools.blinkfinder(combined)
    interval_all = [time_control, time_npc, time_noise, time_secondTask, time_combined]
    f_all = [f_control, f_npc, f_noise, f_secondTask, f_combined]
    interval.append(interval_all)
    frequency.append(f_all)

# Normalization of the eye blink frequency
frequency_ = statistic_tools.normalization(frequency)
frequency = np.array(frequency_)

# ANOVA and post-test (Tukey's method here)
x_labels = ['Control', 'NPC', 'Noise', 'Second Task', 'Combined']
dataF_box, dataF_in = statistic_tools.anova_(frequency, x_labels)
out_list, out_values = statistic_tools.tukey_(dataF_in)

# Plot the violin-plot
plt.figure(1)
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
sns.violinplot(frequency)
sns.swarmplot(dataF_box, edgecolor='w', linewidth=1)
plt.xticks([0, 1, 2, 3, 4], x_labels)
plt.xlabel('Experiments')
plt.ylabel('Blink Frequency (occurrence/minutes)')
plt.plot()

# Draw the adjusted p matrix
p_ = np.zeros((5, 5))
for line in out_list:
    p_[int(line[1]), int(line[0])] = float(line[5])

plt.figure(2)
sns.set_theme(style="white")
mask = np.triu(np.ones_like(p_, dtype=bool))
# cmap = sns.diverging_palette(200, 20, as_cmap=True, center='dark')
# sns.dark_palette("seagreen")
x_labels = ['Control', 'NPC', 'Noise', 'Second Task', '']
y_labels = ['', 'NPC', 'Noise', 'Second Task', 'Combined']
plot2 = sns.heatmap(p_, mask=mask, cmap="magma", vmax=0.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
plot2.set_xticklabels(x_labels, rotation=90)
plot2.set_yticklabels(y_labels, rotation=0)
plt.plot()
