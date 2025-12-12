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


def figure(dataset_Ant_adaorder, dataset_Walker2d_adaorder, dataset_Hopper_order, dataset_Ant_order,
           dataset_Walker2d_order, dataset_Hopper_diffs, dataset_Ant_diffs, dataset_Walker2d_diffs):
    fig = plt.figure(figsize=(13, 7))
    # [左, 下, 宽, 高] 规定的矩形区域 （全部是0~1之间的数，表示比例）
    rect1 = [0.05, 0.65, 0.18, 0.33]
    rect2 = [0.3, 0.65, 0.18, 0.33]
    rect3 = [0.55, 0.65, 0.18, 0.33]
    rect4 = [0.8, 0.65, 0.18, 0.33]
    rect5 = [0.05, 0.13, 0.18, 0.33]
    rect6 = [0.3, 0.13, 0.18, 0.33]
    rect7 = [0.55, 0.13, 0.18, 0.33]
    rect8 = [0.8, 0.13, 0.18, 0.33]
    ax1 = plt.axes(rect1)
    ax2 = plt.axes(rect2)
    ax3 = plt.axes(rect3)
    ax4 = plt.axes(rect4)
    ax5 = plt.axes(rect5)
    ax6 = plt.axes(rect6)
    ax7 = plt.axes(rect7)
    ax8 = plt.axes(rect8)

    color_adaorder = {
        'AdaODDPG(4)': 'green',
        'AdaODDPG(8)': 'red',
        'AdaODDPG(16)': 'm',
        'AdaODDPG(32)': 'c',
    }

    print("Ant_adao")
    for key in dataset_Ant_adaorder.keys():
        data_list = dataset_Ant_adaorder[key]
        data_combined = pd.concat(data_list, ignore_index=True)
        grouped = data_combined.groupby("TotalEnvInteracts")["Performance"]
        mean_raw = grouped.mean()
        std_raw = grouped.std() * 0.05
        xx = mean_raw.index
        y = np.ones(20)
        x_raw = mean_raw.values
        z = np.ones(len(x_raw))
        smoothed_mean = np.convolve(x_raw, y, 'same') / np.convolve(z, y, 'same')
        ax1.plot(xx, smoothed_mean, linewidth=1.5, label=f"{key}", color=color_adaorder[key])
        print(smoothed_mean[-1])
        ax1.fill_between(xx, smoothed_mean - std_raw.values, smoothed_mean + std_raw.values, color=color_adaorder[key],
                         alpha=0.5)
    ax1.set_ylabel("Average return", fontsize=15)
    ax1.set_xlabel("Steps", fontsize=15)
    ax1.set_xlim(0, 5e5)
    ax1.set_ylim(0, 3500)
    ax1.tick_params(labelsize=15)
    xx = MultipleLocator(2e5)
    ax1.xaxis.set_major_locator(xx)
    yy = MultipleLocator(1000)
    ax1.yaxis.set_major_locator(yy)
    ax1.yaxis.get_major_formatter().set_powerlimits((1, 3))
    ax1.yaxis.get_offset_text().set_fontsize(15)
    ax1.xaxis.set_major_locator(xx)
    ax1.xaxis.get_major_formatter().set_powerlimits((5, 5))
    ax1.xaxis.get_offset_text().set_fontsize(15)
    ax1.set_title(label=r'(a)Different $C$ in Ant', fontsize=15, y=-0.4)
    ax1.grid()
    ax1.legend(fontsize=12.5, loc="upper left", handlelength=1)

    print("Walker2d_adaorder")
    for key in dataset_Walker2d_adaorder.keys():
        data_list = dataset_Walker2d_adaorder[key]
        data_combined = pd.concat(data_list, ignore_index=True)
        grouped = data_combined.groupby("TotalEnvInteracts")["Performance"]
        mean_raw = grouped.mean()
        std_raw = grouped.std() * 0.05
        xx = mean_raw.index
        y = np.ones(20)
        x_raw = mean_raw.values
        z = np.ones(len(x_raw))
        smoothed_mean = np.convolve(x_raw, y, 'same') / np.convolve(z, y, 'same')
        ax2.plot(xx, smoothed_mean, linewidth=1.5, label=f"{key}", color=color_adaorder[key])
        print(smoothed_mean[-1])
        ax2.fill_between(xx, smoothed_mean - std_raw.values, smoothed_mean + std_raw.values, color=color_adaorder[key],
                         alpha=0.5)
    ax2.set_ylabel("Average return", fontsize=15)
    ax2.set_xlabel("Steps", fontsize=15)
    ax2.set_xlim(0, 1e6)
    ax2.set_ylim(0, 5200)
    ax2.tick_params(labelsize=15)
    xx = MultipleLocator(3e5)
    ax2.xaxis.set_major_locator(xx)
    yy = MultipleLocator(2000)
    ax2.yaxis.set_major_locator(yy)
    ax2.yaxis.get_major_formatter().set_powerlimits((1, 3))
    ax2.yaxis.get_offset_text().set_fontsize(15)
    ax2.xaxis.set_major_locator(xx)
    ax2.xaxis.get_major_formatter().set_powerlimits((5, 5))
    ax2.xaxis.get_offset_text().set_fontsize(15)
    ax2.set_title(label=r'(b)Different $C$ in Walk2d', fontsize=15, y=-0.4)
    ax2.grid()
    ax2.legend(fontsize=12.5, loc="upper left", handlelength=1)


    color_order = {
        'ODDPG(8,1)': 'green',
        'ODDPG(8,2)': 'c',
        'ODDPG(8,4)': 'm',
        'ODDPG(8,8)': 'y',
        'AdaODDPG(8)': 'red'
    }

    print("Hopper_order")
    for key in dataset_Hopper_order.keys():
        data_list = dataset_Hopper_order[key]
        data_combined = pd.concat(data_list, ignore_index=True)
        grouped = data_combined.groupby("TotalEnvInteracts")["Performance"]
        mean_raw = grouped.mean()
        std_raw = grouped.std() * 0.05
        xx = mean_raw.index
        y = np.ones(20)
        x_raw = mean_raw.values
        z = np.ones(len(x_raw))
        smoothed_mean = np.convolve(x_raw, y, 'same') / np.convolve(z, y, 'same')
        ax3.plot(xx, smoothed_mean, linewidth=1.5, label=f"{key}", color=color_order[key])
        print(smoothed_mean[-1])
        ax3.fill_between(xx, smoothed_mean - std_raw.values, smoothed_mean + std_raw.values, color=color_order[key],
                         alpha=0.5)
    ax3.set_ylabel("Average return", fontsize=15)
    ax3.set_xlabel("Steps", fontsize=15)
    ax3.set_xlim(0, 3e5)
    ax3.set_ylim(0, 3000)
    ax3.tick_params(labelsize=15)
    xx = MultipleLocator(1e5)
    ax3.xaxis.set_major_locator(xx)
    yy = MultipleLocator(1000)
    ax3.yaxis.set_major_locator(yy)
    ax3.yaxis.get_major_formatter().set_powerlimits((1, 3))
    ax3.yaxis.get_offset_text().set_fontsize(15)
    ax3.xaxis.set_major_locator(xx)
    ax3.xaxis.get_major_formatter().set_powerlimits((5, 5))
    ax3.xaxis.get_offset_text().set_fontsize(15)
    ax3.set_title(label=r'(c)Order DDPG in Hopper', fontsize=15, y=-0.4)
    ax3.grid()
    ax3.legend(fontsize=12.5, loc="upper left", handlelength=1)

    print("Ant_order")
    for key in dataset_Ant_order.keys():
        data_list = dataset_Ant_order[key]
        data_combined = pd.concat(data_list, ignore_index=True)
        grouped = data_combined.groupby("TotalEnvInteracts")["Performance"]
        mean_raw = grouped.mean()
        std_raw = grouped.std() * 0.05
        xx = mean_raw.index
        y = np.ones(20)
        x_raw = mean_raw.values
        z = np.ones(len(x_raw))
        smoothed_mean = np.convolve(x_raw, y, 'same') / np.convolve(z, y, 'same')
        ax4.plot(xx, smoothed_mean, linewidth=1.5, label=f"{key}", color=color_order[key])
        print(smoothed_mean[-1])
        ax4.fill_between(xx, smoothed_mean - std_raw.values, smoothed_mean + std_raw.values, color=color_order[key],
                         alpha=0.5)
    ax4.set_ylabel("Average return", fontsize=15)
    ax4.set_xlabel("Steps", fontsize=15)
    ax4.set_xlim(0, 5e5)
    ax4.set_ylim(-500, 3500)
    ax4.tick_params(labelsize=15)
    xx = MultipleLocator(2e5)
    ax4.xaxis.set_major_locator(xx)
    yy = MultipleLocator(1000)
    ax4.yaxis.set_major_locator(yy)
    ax4.yaxis.get_major_formatter().set_powerlimits((1, 3))
    ax4.yaxis.get_offset_text().set_fontsize(15)
    ax4.xaxis.set_major_locator(xx)
    ax4.xaxis.get_major_formatter().set_powerlimits((5, 5))
    ax4.xaxis.get_offset_text().set_fontsize(15)
    ax4.set_title(label=r'(d)Order DDPG in Ant', fontsize=15, y=-0.4)
    ax4.grid()
    ax4.legend(fontsize=12.5, loc="upper left", handlelength=1)

    print("Walker2d_order")
    for key in dataset_Walker2d_order.keys():
        data_list = dataset_Walker2d_order[key]
        data_combined = pd.concat(data_list, ignore_index=True)
        grouped = data_combined.groupby("TotalEnvInteracts")["Performance"]
        mean_raw = grouped.mean()
        std_raw = grouped.std() * 0.05
        xx = mean_raw.index
        y = np.ones(20)
        x_raw = mean_raw.values
        z = np.ones(len(x_raw))
        smoothed_mean = np.convolve(x_raw, y, 'same') / np.convolve(z, y, 'same')
        ax5.plot(xx, smoothed_mean, linewidth=1.5, label=f"{key}", color=color_order[key])
        print(smoothed_mean[-1])
        ax5.fill_between(xx, smoothed_mean - std_raw.values, smoothed_mean + std_raw.values, color=color_order[key],
                         alpha=0.5)
    ax5.set_ylabel("Average return", fontsize=15)
    ax5.set_xlabel("Steps", fontsize=15)
    ax5.set_xlim(0, 1e6)
    ax5.set_ylim(0, 5200)
    ax5.tick_params(labelsize=15)
    xx = MultipleLocator(3e5)
    ax5.xaxis.set_major_locator(xx)
    yy = MultipleLocator(2000)
    ax5.yaxis.set_major_locator(yy)
    ax5.yaxis.get_major_formatter().set_powerlimits((1, 3))
    ax5.yaxis.get_offset_text().set_fontsize(15)
    ax5.xaxis.set_major_locator(xx)
    ax5.xaxis.get_major_formatter().set_powerlimits((5, 5))
    ax5.xaxis.get_offset_text().set_fontsize(15)
    ax5.set_title(label=r'(e)Comparison in Walker2d', fontsize=15, y=-0.4)
    ax5.grid()
    ax5.legend(fontsize=12.5, loc="upper left", handlelength=1)



    color_operation= {
        'AdaADDPG(8)': 'green',
        'AdaMDDPG(8)': 'c',
        'AdaODDPG(8)': 'red'
    }

    print("Hopper_diff")
    for key in dataset_Hopper_diffs.keys():
        data_list = dataset_Hopper_diffs[key]
        data_combined = pd.concat(data_list, ignore_index=True)
        grouped = data_combined.groupby("TotalEnvInteracts")["Performance"]
        mean_raw = grouped.mean()
        std_raw = grouped.std() * 0.05
        xx = mean_raw.index
        y = np.ones(20)
        x_raw = mean_raw.values
        z = np.ones(len(x_raw))
        smoothed_mean = np.convolve(x_raw, y, 'same') / np.convolve(z, y, 'same')
        ax6.plot(xx, smoothed_mean, linewidth=1.5, label=f"{key}", color=color_operation[key])
        print(smoothed_mean[-1])
        ax6.fill_between(xx, smoothed_mean - std_raw.values, smoothed_mean + std_raw.values, color=color_operation[key],
                         alpha=0.5)
    ax6.set_ylabel("Average return", fontsize=15)
    ax6.set_xlabel("Steps", fontsize=15)
    ax6.set_xlim(0, 3e5)
    ax6.set_ylim(0, 3000)
    ax6.tick_params(labelsize=15)
    xx = MultipleLocator(1e5)
    ax6.xaxis.set_major_locator(xx)
    yy = MultipleLocator(1000)
    ax6.yaxis.set_major_locator(yy)
    ax6.yaxis.get_major_formatter().set_powerlimits((1, 3))
    ax6.yaxis.get_offset_text().set_fontsize(15)
    ax6.xaxis.set_major_locator(xx)
    ax6.xaxis.get_major_formatter().set_powerlimits((5, 5))
    ax6.xaxis.get_offset_text().set_fontsize(15)
    ax6.set_title(label=r'(f)Ada-methods in Hopper', fontsize=15, y=-0.4)
    ax6.grid()
    ax6.legend(fontsize=12.5, loc="upper left", handlelength=1)

    print("Ant_diff")
    for key in dataset_Ant_diffs.keys():
        data_list = dataset_Ant_diffs[key]
        data_combined = pd.concat(data_list, ignore_index=True)
        grouped = data_combined.groupby("TotalEnvInteracts")["Performance"]
        mean_raw = grouped.mean()
        std_raw = grouped.std() * 0.05
        xx = mean_raw.index
        y = np.ones(20)
        x_raw = mean_raw.values
        z = np.ones(len(x_raw))
        smoothed_mean = np.convolve(x_raw, y, 'same') / np.convolve(z, y, 'same')
        ax7.plot(xx, smoothed_mean, linewidth=1.5, label=f"{key}", color=color_operation[key])
        print(smoothed_mean[-1])
        ax7.fill_between(xx, smoothed_mean - std_raw.values, smoothed_mean + std_raw.values, color=color_operation[key],
                         alpha=0.5)
    ax7.set_ylabel("Average return", fontsize=15)
    ax7.set_xlabel("Steps", fontsize=15)
    ax7.set_xlim(0, 5e5)
    ax7.set_ylim(0, 3500)
    ax7.tick_params(labelsize=15)
    xx = MultipleLocator(2e5)
    ax7.xaxis.set_major_locator(xx)
    yy = MultipleLocator(1000)
    ax7.yaxis.set_major_locator(yy)
    ax7.yaxis.get_major_formatter().set_powerlimits((1, 3))
    ax7.yaxis.get_offset_text().set_fontsize(15)
    ax7.xaxis.set_major_locator(xx)
    ax7.xaxis.get_major_formatter().set_powerlimits((5, 5))
    ax7.xaxis.get_offset_text().set_fontsize(15)
    ax7.set_title(label=r'(g)Ada-methods in Ant', fontsize=15, y=-0.4)
    ax7.grid()
    ax7.legend(fontsize=12.5, loc="upper left", handlelength=1)

    print("Walker2d_diffs")
    for key in dataset_Walker2d_diffs.keys():
        data_list = dataset_Walker2d_diffs[key]
        data_combined = pd.concat(data_list, ignore_index=True)
        grouped = data_combined.groupby("TotalEnvInteracts")["Performance"]
        mean_raw = grouped.mean()
        std_raw = grouped.std() * 0.05
        xx = mean_raw.index
        y = np.ones(20)
        x_raw = mean_raw.values
        z = np.ones(len(x_raw))
        smoothed_mean = np.convolve(x_raw, y, 'same') / np.convolve(z, y, 'same')
        ax8.plot(xx, smoothed_mean, linewidth=1.5, label=f"{key}", color=color_operation[key])
        print(smoothed_mean[-1])
        ax8.fill_between(xx, smoothed_mean - std_raw.values, smoothed_mean + std_raw.values, color=color_operation[key],
                         alpha=0.5)
    ax8.set_ylabel("Average return", fontsize=15)
    ax8.set_xlabel("Steps", fontsize=15)
    ax8.set_xlim(0, 1e6)
    ax8.set_ylim(0, 5200)
    ax8.tick_params(labelsize=15)
    xx = MultipleLocator(3e5)
    ax8.xaxis.set_major_locator(xx)
    yy = MultipleLocator(2000)
    ax8.yaxis.set_major_locator(yy)
    ax8.yaxis.get_major_formatter().set_powerlimits((1, 3))
    ax8.yaxis.get_offset_text().set_fontsize(15)
    ax8.xaxis.set_major_locator(xx)
    ax8.xaxis.get_major_formatter().set_powerlimits((5, 5))
    ax8.xaxis.get_offset_text().set_fontsize(15)
    ax8.set_title(label=r'(h)Ada-methods in Walker2d', fontsize=15, y=-0.4)
    ax8.grid()
    ax8.legend(fontsize=12.5, loc="upper left", handlelength=1)


    plt.savefig("./SuppleContinualQ.png", dpi=600, bbox_inches='tight', format='png')
    plt.show()



if __name__ == "__main__":
    base_path_Ant_adaorder = '../data/Ant'
    algo_path_Ant_adaorder = {
        'AdaODDPG(4)': 'Ant-v2_adaorder_C4',
        'AdaODDPG(8)': 'Ant-v2_adaorder_C8',
        'AdaODDPG(16)': 'Ant-v2_adaorder_C16',
        'AdaODDPG(32)': 'Ant-v2_adaorder_C32',
    }
    dataset_Ant_adaorder = {}
    dataset_Ant_adaorder.update(get_dataset(algo_path_Ant_adaorder, base_path_Ant_adaorder))

    base_path_Walker2d_adaorder = '../data/Walker2d'
    algo_path_Walker2d_adaorder = {
        'AdaODDPG(4)': 'Walker2d-v2_adaorder_C4',
        'AdaODDPG(8)': 'Walker2d-v2_adaorder_C8',
        'AdaODDPG(16)': 'Walker2d-v2_adaorder_C16',
        'AdaODDPG(32)': 'Walker2d-v2_adaorder_C32',
    }
    dataset_Walker2d_adaorder = {}
    dataset_Walker2d_adaorder.update(get_dataset(algo_path_Walker2d_adaorder, base_path_Walker2d_adaorder))

    base_path_Hopper_order = '../data/Hopper'
    algo_path_Hopper_order = {
        'ODDPG(8,1)': 'Hopper-v2_maxmin_N8',
        'ODDPG(8,2)': 'Hopper-v2_order_N8_O2',
        'ODDPG(8,4)': 'Hopper-v2_order_N8_O4',
        'ODDPG(8,8)': 'Hopper-v2_order_N8_O8',
        'AdaODDPG(8)': 'Hopper-v2_adaorder_C8',
    }
    dataset_Hopper_order = {}
    dataset_Hopper_order.update(get_dataset(algo_path_Hopper_order, base_path_Hopper_order))

    base_path_Ant_order = '../data/Ant'
    algo_path_Ant_order = {
        'ODDPG(8,1)': 'Ant-v2_maxmin_N8',
        'ODDPG(8,2)': 'Ant-v2_order_N8_O2',
        'ODDPG(8,4)': 'Ant-v2_order_N8_O4',
        'ODDPG(8,8)': 'Ant-v2_order_N8_O8',
        'AdaODDPG(8)': 'Ant-v2_adaorder_C8',
    }
    dataset_Ant_order = {}
    dataset_Ant_order.update(get_dataset(algo_path_Ant_order, base_path_Ant_order))

    base_path_Walker2d_order = '../data/Walker2d'
    algo_path_Walker2d_order = {
        'ODDPG(8,1)': 'Walker2d-v2_maxmin_N8',
        'ODDPG(8,2)': 'Walker2d-v2_order_N8_O2',
        'ODDPG(8,4)': 'Walker2d-v2_order_N8_O4',
        'ODDPG(8,8)': 'Walker2d-v2_order_N8_O8',
        'AdaODDPG(8)': 'Walker2d-v2_adaorder_C8',
    }
    dataset_Walker2d_order = {}
    dataset_Walker2d_order.update(get_dataset(algo_path_Walker2d_order, base_path_Walker2d_order))

    base_path_Hopper_diffs = '../data/Hopper'
    algo_path_Hopper_diffs = {
        'AdaADDPG(8)': 'Hopper-v2_adaaverage_C8',
        'AdaMDDPG(8)': 'Hopper-v2_adamin_C8',
        'AdaODDPG(8)': 'Hopper-v2_adaorder_C8'
    }
    dataset_Hopper_diffs = {}
    dataset_Hopper_diffs.update(get_dataset(algo_path_Hopper_diffs, base_path_Hopper_diffs))

    base_path_Ant_diffs = '../data/Ant'
    algo_path_Ant_diffs = {
        'AdaADDPG(8)': 'Ant-v2_adaaverage_C8',
        'AdaMDDPG(8)': 'Ant-v2_adamin_C8',
        'AdaODDPG(8)': 'Ant-v2_adaorder_C8',
    }
    dataset_Ant_diffs = {}
    dataset_Ant_diffs.update(get_dataset(algo_path_Ant_diffs, base_path_Ant_diffs))

    base_path_Walker2d_diffs = '../data/Walker2d'
    algo_path_Walker2d_diffs = {
        'AdaADDPG(8)': 'Walker2d-v2_adaaverage_C8',
        'AdaMDDPG(8)': 'Walker2d-v2_adamin_C8',
        'AdaODDPG(8)': 'Walker2d-v2_adaorder_C8'
    }
    dataset_Walker2d_diffs = {}
    dataset_Walker2d_diffs.update(get_dataset(algo_path_Walker2d_diffs, base_path_Walker2d_diffs))

    figure(dataset_Ant_adaorder, dataset_Walker2d_adaorder, dataset_Hopper_order, dataset_Ant_order,
           dataset_Walker2d_order, dataset_Hopper_diffs, dataset_Ant_diffs, dataset_Walker2d_diffs)

