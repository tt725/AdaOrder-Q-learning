from utils.plotter import Plotter
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


def get_date(env_name, x_label, y_label, indexList, runs):
    plotter = Plotter(env_name=env_name, merged=True, x_label=x_label, y_label=y_label,
                      ci="se", EMA=True, runs=runs)
    data = plotter.result_indexList(indexList, mode='Train')
    return data


def figure(copter_Order_sota, Breakout_Order_sota, asterix_Order_sota, copter_Order_ada):
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

    label_sota = [r'ADQN',
                  r"MDQN",
                  r'SCDQN',
                  r'SDQN',
                  r'WDDQN',
                  r'REDQ',
                  r'EBDL',
                  r"AdaEQ",
                  r'AdaODQN']

    color_sota = ['blue',
                  'black',
                  'green',
                  'c',
                  'm',
                  'y',
                  '#1f77b4',
                  '#ff7f0e',
                  'red']
    print("copter sota")
    for i in range(len(copter_Order_sota)):
        ys = []
        for result in copter_Order_sota[i]:
            ys.append(result[y_label].to_numpy())
        ys = np.array(ys)
        x_mean = copter_Order_sota[i][0][x_label].to_numpy()
        y_mean = np.mean(ys, axis=0)
        y_ci = np.std(ys, axis=0, ddof=0) / math.sqrt(len(ys))
        ax1.plot(x_mean, y_mean, linewidth=1.5, label=label_sota[i], color=color_sota[i])
        ax1.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, alpha=0.5)
        print(y_mean[-1])
    ax1.set_ylabel("Average score per episode", fontsize=15)
    ax1.set_xlabel("Frames", fontsize=15)
    ax1.set_xlim(0, 1e6)
    ax1.set_ylim(0, 40)
    ax1.tick_params(labelsize=15)
    xx = MultipleLocator(4e5)
    ax1.xaxis.set_major_locator(xx)
    yy = MultipleLocator(20)
    ax1.yaxis.set_major_locator(yy)
    ax1.yaxis.get_major_formatter().set_powerlimits((1, 1))
    ax1.yaxis.get_offset_text().set_fontsize(15)
    ax1.xaxis.set_major_locator(xx)
    ax1.xaxis.get_major_formatter().set_powerlimits((5, 5))
    ax1.xaxis.get_offset_text().set_fontsize(15)
    ax1.set_title(label=r'(a)Comparison in Pixelcopter', fontsize=15, y=-0.4)
    ax1.grid()
    ax1.legend(fontsize=12.5, loc="upper left", handlelength=1)

    print("breakout sota")
    for i in range(len(Breakout_Order_sota)):
        ys = []
        for result in Breakout_Order_sota[i]:
            ys.append(result[y_label].to_numpy())
        ys = np.array(ys)
        x_mean = Breakout_Order_sota[i][0][x_label].to_numpy()
        y_mean = np.mean(ys, axis=0)
        y_ci = np.std(ys, axis=0, ddof=0) / math.sqrt(len(ys))
        ax2.plot(x_mean, y_mean, linewidth=1.5, label=label_sota[i], color=color_sota[i])
        ax2.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, alpha=0.5)
        print(y_mean[-1])
    ax2.set_ylabel("Average score per episode", fontsize=15)
    ax2.set_xlabel("Frames", fontsize=15)
    ax2.set_xlim(0, 1e6)
    ax2.set_ylim(0, 11)
    ax2.tick_params(labelsize=15)
    xx = MultipleLocator(4e5)
    ax2.xaxis.set_major_locator(xx)
    yy = MultipleLocator(4)
    ax2.yaxis.set_major_locator(yy)
    ax2.yaxis.get_offset_text().set_fontsize(15)
    ax2.xaxis.set_major_locator(xx)
    ax2.xaxis.get_major_formatter().set_powerlimits((5, 5))
    ax2.xaxis.get_offset_text().set_fontsize(15)
    ax2.set_title(label=r'(b)Comparison in Breakout', fontsize=15, y=-0.4)
    ax2.grid()
    ax2.legend(fontsize=12.5, loc="upper left", handlelength=1)

    print("asterix sota")
    for i in range(len(asterix_Order_sota)):
        ys = []
        for result in asterix_Order_sota[i]:
            ys.append(result[y_label].to_numpy())
        ys = np.array(ys)
        x_mean = asterix_Order_sota[i][0][x_label].to_numpy()
        y_mean = np.mean(ys, axis=0)
        y_ci = np.std(ys, axis=0, ddof=0) / math.sqrt(len(ys))
        ax3.plot(x_mean, y_mean, linewidth=1.5, label=label_sota[i], color=color_sota[i])
        ax3.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, alpha=0.5)
        print(y_mean[-1])
    ax3.set_ylabel("Average score per episode", fontsize=15)
    ax3.set_xlabel("Frames", fontsize=15)
    ax3.set_xlim(0, 3e6)
    ax3.set_ylim(0, 20)
    ax3.tick_params(labelsize=15)
    xx = MultipleLocator(1e6)
    ax3.xaxis.set_major_locator(xx)
    yy = MultipleLocator(10)
    ax3.yaxis.set_major_locator(yy)
    ax3.yaxis.get_major_formatter().set_powerlimits((1, 1))
    ax3.yaxis.get_offset_text().set_fontsize(15)
    ax3.xaxis.set_major_locator(xx)
    ax3.xaxis.get_major_formatter().set_powerlimits((6, 6))
    ax3.xaxis.get_offset_text().set_fontsize(15)
    ax3.set_title(label=r'(c)Comparison in Asterix', fontsize=15, y=-0.4)
    ax3.grid()
    ax3.legend(fontsize=12.5, loc="upper left", handlelength=1)

    label_ada = [r'DQN',
                  r"DDQN",
                  r'AdaODQN(4)',
                  r'AdaODQN(8)',
                  r'AdaODQN(16)',
                  r'AdaODQN(32)']

    color_ada = ['blue',
                  'black',
                  'green',
                  'red',
                  'm',
                  'c']
    print("copter ada")
    for i in range(len(copter_Order_ada)):
        ys = []
        for result in copter_Order_ada[i]:
            ys.append(result[y_label].to_numpy())
        ys = np.array(ys)
        x_mean = copter_Order_ada[i][0][x_label].to_numpy()
        y_mean = np.mean(ys, axis=0)
        y_ci = np.std(ys, axis=0, ddof=0) / math.sqrt(len(ys))
        ax4.plot(x_mean, y_mean, linewidth=1.5, label=label_ada[i], color=color_ada[i])
        ax4.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, alpha=0.5)
        print(y_mean[-1])
    ax4.set_ylabel("Average score per episode", fontsize=15)
    ax4.set_xlabel("Frames", fontsize=15)
    ax4.set_xlim(0, 1e6)
    ax4.set_ylim(0, 40)
    ax4.tick_params(labelsize=15)
    xx = MultipleLocator(4e5)
    ax4.xaxis.set_major_locator(xx)
    yy = MultipleLocator(20)
    ax4.yaxis.set_major_locator(yy)
    ax4.yaxis.get_major_formatter().set_powerlimits((1, 1))
    ax4.yaxis.get_offset_text().set_fontsize(15)
    ax4.xaxis.set_major_locator(xx)
    ax4.xaxis.get_major_formatter().set_powerlimits((5, 5))
    ax4.xaxis.get_offset_text().set_fontsize(15)
    ax4.set_title(label=r'(d)Different $C$ in Pixelcopter', fontsize=15, y=-0.4)
    ax4.grid()
    ax4.legend(fontsize=12.5, loc="upper left", handlelength=1)

    plt.savefig("./DeepQ.png", dpi=600, bbox_inches='tight', format='png')
    plt.show()


if __name__ == "__main__":
    x_label = 'Step'
    y_label = 'Average Return'
    copter_Order_sota = get_date(env_name='copter', x_label=x_label, y_label=y_label,
                                 indexList=[7, 6, 4, 5, 3, 8, 9, 10, 17],
                                 runs=20)
    Breakout_Order_sota = get_date(env_name='minatar_Breakout', x_label=x_label, y_label=y_label,
                                   indexList=[7, 6, 4, 5, 3, 8, 9, 10, 17],
                                   runs=20)
    asterix_Order_sota = get_date(env_name='minatar_Asterix', x_label=x_label, y_label=y_label,
                                  indexList=[7, 6, 4, 5, 3, 8, 9, 10, 17],
                                  runs=20)
    copter_Order_ada = get_date(env_name='copter', x_label=x_label, y_label=y_label,
                                indexList=[1, 2, 16, 17, 18, 19],
                                runs=20)
    figure(copter_Order_sota, Breakout_Order_sota, asterix_Order_sota, copter_Order_ada)
