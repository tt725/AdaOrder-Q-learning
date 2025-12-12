import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import pandas as pd

exp_idx = 0
units = dict()
def get_dataset(algo_path, base_path):
    global exp_idx
    global units

    dataset = {}
    for key in algo_path.keys():
        complete_path = os.path.join(base_path, algo_path[key])
        temp_data = []
        for root, _, files in os.walk(complete_path):
            if 'progress.txt' in files:
                data = pd.read_table(os.path.join(root, 'progress.txt'))

                condition = key
                exp_idx += 1
                if condition not in units:
                    units[condition] = 0
                unit = units[condition]
                units[condition] += 1

                performance = 'AverageTestEpRet' if 'AverageTestEpRet' in data else 'AverageEpRet'
                data.insert(len(data.columns), 'Unit', unit)
                data.insert(len(data.columns), 'Condition', condition)
                if performance in data:
                    data.insert(len(data.columns), 'Performance', data[performance])
                temp_data.append(data)
        dataset[key] = temp_data
    exp_idx = 0
    units = dict()
    return dataset


def figure(dataset_Hopper, dataset_Ant, dataset_Walker2d, dataset_Hopper_adaorder):
    fig = plt.figure(figsize=(13, 3.5))

    # [左, 下, 宽, 高] 规定的矩形区域 （全部是0~1之间的数，表示比例）
    rect1 = [0.05, 0.26, 0.18, 0.68]
    rect2 = [0.3, 0.26, 0.18, 0.68]
    rect3 = [0.55, 0.26, 0.18, 0.68]
    rect4 = [0.8, 0.26, 0.18, 0.68]
    ax1 = plt.axes(rect1)
    ax2 = plt.axes(rect2)
    ax3 = plt.axes(rect3)
    ax4 = plt.axes(rect4)


    color_sota = {
        'DDPG': 'blue',
        'AQ': 'black',
        'MQ': 'green',
        'REDQ': 'yellow',
        'AdaEQ': 'brown',
        'AdaODDPG': 'red',
    }

    print("Hopper sota")
    for key in dataset_Hopper.keys():
        data_list = dataset_Hopper[key]
        data_combined = pd.concat(data_list, ignore_index=True)
        grouped = data_combined.groupby("TotalEnvInteracts")["Performance"]
        mean_raw = grouped.mean()
        std_raw = grouped.std() * 0.05
        xx = mean_raw.index
        y = np.ones(20)
        x_raw = mean_raw.values
        z = np.ones(len(x_raw))
        smoothed_mean = np.convolve(x_raw, y, 'same') / np.convolve(z, y, 'same')
        ax1.plot(xx, smoothed_mean, linewidth=1.5, label=f"{key}", color=color_sota[key])
        print(smoothed_mean[-1])
        ax1.fill_between(xx, smoothed_mean - std_raw.values, smoothed_mean + std_raw.values, color=color_sota[key],
                         alpha=0.5)

    ax1.set_ylabel("Average return", fontsize=15)
    ax1.set_xlabel("Steps", fontsize=15)
    ax1.set_xlim(0, 3e5)
    ax1.set_ylim(0, 3000)
    ax1.tick_params(labelsize=15)
    xx = MultipleLocator(1e5)
    ax1.xaxis.set_major_locator(xx)
    yy = MultipleLocator(1000)
    ax1.yaxis.set_major_locator(yy)
    ax1.yaxis.get_major_formatter().set_powerlimits((1, 3))
    ax1.yaxis.get_offset_text().set_fontsize(15)
    ax1.xaxis.set_major_locator(xx)
    ax1.xaxis.get_major_formatter().set_powerlimits((5, 5))
    ax1.xaxis.get_offset_text().set_fontsize(15)
    ax1.set_title(label=r'(a)Comparison in Hopper', fontsize=15, y=-0.4)
    ax1.grid()
    ax1.legend(fontsize=12.5, loc="upper left", handlelength=1)

    print("Ant sota")
    for key in dataset_Ant.keys():
        data_list = dataset_Ant[key]
        data_combined = pd.concat(data_list, ignore_index=True)
        grouped = data_combined.groupby("TotalEnvInteracts")["Performance"]
        mean_raw = grouped.mean()
        std_raw = grouped.std() * 0.05
        xx = mean_raw.index
        y = np.ones(20)
        x_raw = mean_raw.values
        z = np.ones(len(x_raw))
        smoothed_mean = np.convolve(x_raw, y, 'same') / np.convolve(z, y, 'same')
        ax2.plot(xx, smoothed_mean, linewidth=1.5, label=f"{key}", color=color_sota[key])
        print(smoothed_mean[-1])
        ax2.fill_between(xx, smoothed_mean - std_raw.values, smoothed_mean + std_raw.values, color=color_sota[key],
                         alpha=0.5)
    ax2.set_ylabel("Average return", fontsize=15)
    ax2.set_xlabel("Steps", fontsize=15)
    ax2.set_xlim(0, 5e5)
    ax2.set_ylim(0, 3500)
    ax2.tick_params(labelsize=15)
    xx = MultipleLocator(2e5)
    ax2.xaxis.set_major_locator(xx)
    yy = MultipleLocator(1000)
    ax2.yaxis.set_major_locator(yy)
    ax2.yaxis.get_major_formatter().set_powerlimits((1, 3))
    ax2.yaxis.get_offset_text().set_fontsize(15)
    ax2.xaxis.set_major_locator(xx)
    ax2.xaxis.get_major_formatter().set_powerlimits((5, 5))
    ax2.xaxis.get_offset_text().set_fontsize(15)
    ax2.set_title(label=r'(b)Comparison in Ant', fontsize=15, y=-0.4)
    ax2.grid()
    ax2.legend(fontsize=12.5, loc="upper left", handlelength=1)

    print("Walker2d sota")
    for key in dataset_Walker2d.keys():
        data_list = dataset_Walker2d[key]
        data_combined = pd.concat(data_list, ignore_index=True)
        grouped = data_combined.groupby("TotalEnvInteracts")["Performance"]
        mean_raw = grouped.mean()
        std_raw = grouped.std() * 0.05
        xx = mean_raw.index
        y = np.ones(20)
        x_raw = mean_raw.values
        z = np.ones(len(x_raw))
        smoothed_mean = np.convolve(x_raw, y, 'same') / np.convolve(z, y, 'same')
        ax3.plot(xx, smoothed_mean, linewidth=1.5, label=f"{key}", color=color_sota[key])
        print(smoothed_mean[-1])
        ax3.fill_between(xx, smoothed_mean - std_raw.values, smoothed_mean + std_raw.values, color=color_sota[key],
                         alpha=0.5)
    ax3.set_ylabel("Average return", fontsize=15)
    ax3.set_xlabel("Steps", fontsize=15)
    ax3.set_xlim(0, 1e6)
    ax3.set_ylim(0, 5200)
    ax3.tick_params(labelsize=15)
    xx = MultipleLocator(3e5)
    ax3.xaxis.set_major_locator(xx)
    yy = MultipleLocator(2000)
    ax3.yaxis.set_major_locator(yy)
    ax3.yaxis.get_major_formatter().set_powerlimits((1, 3))
    ax3.yaxis.get_offset_text().set_fontsize(15)
    ax3.xaxis.set_major_locator(xx)
    ax3.xaxis.get_major_formatter().set_powerlimits((5, 5))
    ax3.xaxis.get_offset_text().set_fontsize(15)
    ax3.set_title(label=r'(c)Comparison in Walker2d', fontsize=15, y=-0.4)
    ax3.grid()
    ax3.legend(fontsize=12.5, loc="upper left", handlelength=1)

    color_adaorder = {
        'AdaODDPG(4)': 'green',
        'AdaODDPG(8)': 'red',
        'AdaODDPG(16)': 'm',
        'AdaODDPG(32)': 'c',
    }

    print("Hopper_adaorder")
    for key in dataset_Hopper_adaorder.keys():
        data_list = dataset_Hopper_adaorder[key]
        data_combined = pd.concat(data_list, ignore_index=True)
        grouped = data_combined.groupby("TotalEnvInteracts")["Performance"]
        mean_raw = grouped.mean()
        std_raw = grouped.std() * 0.05
        xx = mean_raw.index
        y = np.ones(20)
        x_raw = mean_raw.values
        z = np.ones(len(x_raw))
        smoothed_mean = np.convolve(x_raw, y, 'same') / np.convolve(z, y, 'same')
        ax4.plot(xx, smoothed_mean, linewidth=1.5, label=f"{key}", color=color_adaorder[key])
        print(smoothed_mean[-1])
        ax4.fill_between(xx, smoothed_mean - std_raw.values, smoothed_mean + std_raw.values, color=color_adaorder[key],
                         alpha=0.5)
    ax4.set_ylabel("Average return", fontsize=15)
    ax4.set_xlabel("Steps", fontsize=15)
    ax4.set_xlim(0, 3e5)
    ax4.set_ylim(0, 3000)
    ax4.tick_params(labelsize=15)
    xx = MultipleLocator(1e5)
    ax4.xaxis.set_major_locator(xx)
    yy = MultipleLocator(1000)
    ax4.yaxis.set_major_locator(yy)
    ax4.yaxis.get_major_formatter().set_powerlimits((1, 3))
    ax4.yaxis.get_offset_text().set_fontsize(15)
    ax4.xaxis.set_major_locator(xx)
    ax4.xaxis.get_major_formatter().set_powerlimits((5, 5))
    ax4.xaxis.get_offset_text().set_fontsize(15)
    ax4.set_title(label=r'(d)Different $C$ in Hopper', fontsize=15, y=-0.4)
    ax4.grid()
    ax4.legend(fontsize=12.5, loc="upper left", handlelength=1)

    plt.savefig("./ContinualQ.png", dpi=600, bbox_inches='tight', format='png')
    plt.show()


if __name__ == "__main__":
    base_path_Hopper = '../data/Hopper'
    algo_path_Hopper = {
        'DDPG': 'Hopper-v2_ddpg_N1',
        'AQ': 'Hopper-v2_average_N8',
        'MQ': 'Hopper-v2_maxmin_N8',
        'REDQ': 'Hopper-v2_redq_N8_M3',
        'AdaEQ': 'Hopper-v2_adaeq_N8_M3_C0.3',
        'AdaODDPG': 'Hopper-v2_adaorder_C8',
    }
    dataset_Hopper = {}
    dataset_Hopper.update(get_dataset(algo_path_Hopper, base_path_Hopper))

    base_path_Ant = '../data/Ant'
    algo_path_Ant = {
        'DDPG': 'Ant-v2_ddpg_N1',
        'AQ': 'Ant-v2_average_N8',
        'MQ': 'Ant-v2_maxmin_N8',
        'REDQ': 'Ant-v2_redq_N8_M3',
        'AdaEQ': 'Ant-v2_adaeq_N8_M3_C0.3',
        'AdaODDPG': 'Ant-v2_adaorder_C8',
    }
    dataset_Ant = {}
    dataset_Ant.update(get_dataset(algo_path_Ant, base_path_Ant))

    base_path_Walker2d = '../data/Walker2d'
    algo_path_Walker2d = {
        'DDPG': 'Walker2d-v2_ddpg_N1',
        'AQ': 'Walker2d-v2_average_N8',
        'MQ': 'Walker2d-v2_maxmin_N8',
        'REDQ': 'Walker2d-v2_redq_N8_M3',
        'AdaEQ': 'Walker2d-v2_adaeq_N8_M3_C0.3',
        'AdaODDPG': 'Walker2d-v2_adaorder_C8',
    }
    dataset_Walker2d = {}
    dataset_Walker2d.update(get_dataset(algo_path_Walker2d, base_path_Walker2d))

    base_path_Hopper_adaorder = '../data/Hopper'
    algo_path_Hopper_adaorder = {
        'AdaODDPG(4)': 'Hopper-v2_adaorder_C4',
        'AdaODDPG(8)': 'Hopper-v2_adaorder_C8',
        'AdaODDPG(16)': 'Hopper-v2_adaorder_C16',
        'AdaODDPG(32)': 'Hopper-v2_adaorder_C32',
    }
    dataset_Hopper_adaorder = {}
    dataset_Hopper_adaorder.update(get_dataset(algo_path_Hopper_adaorder, base_path_Hopper_adaorder))

    figure(dataset_Hopper, dataset_Ant, dataset_Walker2d, dataset_Hopper_adaorder)
