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


def figure(Breakout_Order_ada, Asterix_Order_ada, copter_Order, Breakout_Order, asterix_Order, copter_comparsion, Breakout_comparsion, asterix_comparsion):
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
    print("Breakout_Order_ada")
    for i in range(len(Breakout_Order_ada)):
        ys = []
        for result in Breakout_Order_ada[i]:
            ys.append(result[y_label].to_numpy())
        ys = np.array(ys)
        x_mean = Breakout_Order_ada[i][0][x_label].to_numpy()
        y_mean = np.mean(ys, axis=0)
        y_ci = np.std(ys, axis=0, ddof=0) / math.sqrt(len(ys))
        ax1.plot(x_mean, y_mean, linewidth=1.0, label=label_ada[i], color=color_ada[i])
        ax1.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, alpha=0.5)
        print(y_mean[-1])
    ax1.set_ylabel("Average score per episode", fontsize=15)
    ax1.set_xlabel("Frames", fontsize=15)
    ax1.set_xlim(0, 1e6)
    ax1.set_ylim(0, 11)
    ax1.tick_params(labelsize=15)
    xx = MultipleLocator(4e5)
    ax1.xaxis.set_major_locator(xx)
    yy = MultipleLocator(4)
    ax1.yaxis.set_major_locator(yy)
    ax1.yaxis.get_offset_text().set_fontsize(15)
    ax1.xaxis.set_major_locator(xx)
    ax1.xaxis.get_major_formatter().set_powerlimits((5, 5))
    ax1.xaxis.get_offset_text().set_fontsize(15)
    ax1.set_title(label=r'(a)Different $C$ in Breakout', fontsize=15, y=-0.4)
    ax1.grid()
    ax1.legend(fontsize=12, loc="upper left", handlelength=1)

    print("Asterix_Order_ada")
    for i in range(len(Asterix_Order_ada)):
        ys = []
        for result in Asterix_Order_ada[i]:
            ys.append(result[y_label].to_numpy())
        ys = np.array(ys)
        x_mean = Asterix_Order_ada[i][0][x_label].to_numpy()
        y_mean = np.mean(ys, axis=0)
        y_ci = np.std(ys, axis=0, ddof=0) / math.sqrt(len(ys))
        ax2.plot(x_mean, y_mean, linewidth=1.0, label=label_ada[i], color=color_ada[i])
        ax2.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, alpha=0.5)
        print(y_mean[-1])
    ax2.set_ylabel("Average score per episode", fontsize=15)
    ax2.set_xlabel("Frames", fontsize=15)
    ax2.set_xlim(0, 3e6)
    ax2.set_ylim(0, 20)
    ax2.tick_params(labelsize=15)
    xx = MultipleLocator(1e6)
    ax2.xaxis.set_major_locator(xx)
    yy = MultipleLocator(10)
    ax2.yaxis.set_major_locator(yy)
    ax2.yaxis.get_major_formatter().set_powerlimits((1, 1))
    ax2.yaxis.get_offset_text().set_fontsize(15)
    ax2.xaxis.set_major_locator(xx)
    ax2.xaxis.get_major_formatter().set_powerlimits((6, 6))
    ax2.xaxis.get_offset_text().set_fontsize(15)
    ax2.set_title(label=r'(b)Different $C$ in Asterix', fontsize=15, y=-0.4)
    ax2.grid()
    ax2.legend(fontsize=12, loc="upper left", handlelength=1)

    label_order = [r'ODQN(8,1)',
                 r'ODQN(8,2)',
                 r'ODQN(8,4)',
                 r'ODQN(8,8)',
                 r'AdaODQN(8)']

    color_order = ['green',
                  'c',
                  'm',
                  'y',
                  'red']
    print("copter_Order")
    for i in range(len(copter_Order)):
        ys = []
        for result in copter_Order[i]:
            ys.append(result[y_label].to_numpy())
        ys = np.array(ys)
        x_mean = copter_Order[i][0][x_label].to_numpy()
        y_mean = np.mean(ys, axis=0)
        y_ci = np.std(ys, axis=0, ddof=0) / math.sqrt(len(ys))
        ax3.plot(x_mean, y_mean, linewidth=1.0, label=label_order[i], color=color_order[i])
        ax3.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, alpha=0.5)
        print(y_mean[-1])
    ax3.set_ylabel("Average score per episode", fontsize=15)
    ax3.set_xlabel("Frames", fontsize=15)
    ax3.set_xlim(0, 1e6)
    ax3.set_ylim(0, 40)
    ax3.tick_params(labelsize=15)
    xx = MultipleLocator(4e5)
    ax3.xaxis.set_major_locator(xx)
    yy = MultipleLocator(20)
    ax3.yaxis.set_major_locator(yy)
    ax3.yaxis.get_major_formatter().set_powerlimits((1, 1))
    ax3.yaxis.get_offset_text().set_fontsize(15)
    ax3.xaxis.set_major_locator(xx)
    ax3.xaxis.get_major_formatter().set_powerlimits((5, 5))
    ax3.xaxis.get_offset_text().set_fontsize(15)
    ax3.set_title(label=r'(c)ODQN in Pixelcopter', fontsize=15, y=-0.4)
    ax3.grid()
    ax3.legend(fontsize=12, loc="upper left", handlelength=1)

    print("Breakout_Order")
    for i in range(len(Breakout_Order)):
        ys = []
        for result in Breakout_Order[i]:
            ys.append(result[y_label].to_numpy())
        ys = np.array(ys)
        x_mean = Breakout_Order[i][0][x_label].to_numpy()
        y_mean = np.mean(ys, axis=0)
        y_ci = np.std(ys, axis=0, ddof=0) / math.sqrt(len(ys))
        ax4.plot(x_mean, y_mean, linewidth=1.0, label=label_order[i], color=color_order[i])
        ax4.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, alpha=0.5)
        print(y_mean[-1])
    ax4.set_ylabel("Average score per episode", fontsize=15)
    ax4.set_xlabel("Frames", fontsize=15)
    ax4.set_xlim(0, 1e6)
    ax4.set_ylim(0, 11)
    ax4.tick_params(labelsize=15)
    xx = MultipleLocator(4e5)
    ax4.xaxis.set_major_locator(xx)
    yy = MultipleLocator(4)
    ax4.yaxis.set_major_locator(yy)
    ax4.yaxis.get_offset_text().set_fontsize(15)
    ax4.xaxis.set_major_locator(xx)
    ax4.xaxis.get_major_formatter().set_powerlimits((5, 5))
    ax4.xaxis.get_offset_text().set_fontsize(15)
    ax4.set_title(label=r'(d)ODQN in Breakout', fontsize=15, y=-0.4)
    ax4.grid()
    ax4.legend(fontsize=12, loc="upper left", handlelength=1)

    print("asterix_Order")
    for i in range(len(asterix_Order)):
        ys = []
        for result in asterix_Order[i]:
            ys.append(result[y_label].to_numpy())
        ys = np.array(ys)
        x_mean = asterix_Order[i][0][x_label].to_numpy()
        y_mean = np.mean(ys, axis=0)
        y_ci = np.std(ys, axis=0, ddof=0) / math.sqrt(len(ys))
        ax5.plot(x_mean, y_mean, linewidth=1.0, label=label_order[i], color=color_order[i])
        ax5.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, alpha=0.5)
        print(y_mean[-1])
    ax5.set_ylabel("Average score per episode", fontsize=15)
    ax5.set_xlabel("Frames", fontsize=15)
    ax5.set_xlim(0, 3e6)
    ax5.set_ylim(0, 20)
    ax5.tick_params(labelsize=15)
    xx = MultipleLocator(1e6)
    ax5.xaxis.set_major_locator(xx)
    yy = MultipleLocator(10)
    ax5.yaxis.set_major_locator(yy)
    ax5.yaxis.get_major_formatter().set_powerlimits((1, 1))
    ax5.yaxis.get_offset_text().set_fontsize(15)
    ax5.xaxis.set_major_locator(xx)
    ax5.xaxis.get_major_formatter().set_powerlimits((6, 6))
    ax5.xaxis.get_offset_text().set_fontsize(15)
    ax5.set_title(label=r'(e)ODQN in Asterix', fontsize=15, y=-0.4)
    ax5.grid()
    ax5.legend(fontsize=12, loc="upper left", handlelength=1)

    label_operation = [r'AdaADQN(8)',
                   r"AdaMDQN(8)",
                   r'AdaODQN(8)']

    color_operation = ['green',
                   'c',
                   'red']
    print("copter_comparsion")
    for i in range(len(copter_comparsion)):
        ys = []
        for result in copter_comparsion[i]:
            ys.append(result[y_label].to_numpy())
        ys = np.array(ys)
        x_mean = copter_comparsion[i][0][x_label].to_numpy()
        y_mean = np.mean(ys, axis=0)
        y_ci = np.std(ys, axis=0, ddof=0) / math.sqrt(len(ys))
        ax6.plot(x_mean, y_mean, linewidth=1.0, label=label_operation[i], color=color_operation[i])
        ax6.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, alpha=0.5)
        print(y_mean[-1])
    ax6.set_ylabel("Average score per episode", fontsize=15)
    ax6.set_xlabel("Frames", fontsize=15)
    ax6.set_xlim(0, 1e6)
    ax6.set_ylim(0, 40)
    ax6.tick_params(labelsize=15)
    xx = MultipleLocator(4e5)
    ax6.xaxis.set_major_locator(xx)
    yy = MultipleLocator(20)
    ax6.yaxis.set_major_locator(yy)
    ax6.yaxis.get_major_formatter().set_powerlimits((1, 1))
    ax6.yaxis.get_offset_text().set_fontsize(15)
    ax6.xaxis.set_major_locator(xx)
    ax6.xaxis.get_major_formatter().set_powerlimits((5, 5))
    ax6.xaxis.get_offset_text().set_fontsize(15)
    ax6.set_title(label=r'(f)Ada-methods in Pixelcopter', fontsize=15, y=-0.4)
    ax6.grid()
    ax6.legend(fontsize=12, loc="upper left", handlelength=1)

    print("Breakout_comparsion")
    for i in range(len(Breakout_comparsion)):
        ys = []
        for result in Breakout_comparsion[i]:
            ys.append(result[y_label].to_numpy())
        ys = np.array(ys)
        x_mean = Breakout_comparsion[i][0][x_label].to_numpy()
        y_mean = np.mean(ys, axis=0)
        y_ci = np.std(ys, axis=0, ddof=0) / math.sqrt(len(ys))
        ax7.plot(x_mean, y_mean, linewidth=1.0, label=label_operation[i], color=color_operation[i])
        ax7.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, alpha=0.5)
        print(y_mean[-1])
    ax7.set_ylabel("Average score per episode", fontsize=15)
    ax7.set_xlabel("Frames", fontsize=15)
    ax7.set_xlim(0, 1e6)
    ax7.set_ylim(0, 11)
    ax7.tick_params(labelsize=15)
    xx = MultipleLocator(4e5)
    ax7.xaxis.set_major_locator(xx)
    yy = MultipleLocator(4)
    ax7.yaxis.set_major_locator(yy)
    ax7.yaxis.get_offset_text().set_fontsize(15)
    ax7.xaxis.set_major_locator(xx)
    ax7.xaxis.get_major_formatter().set_powerlimits((5, 5))
    ax7.xaxis.get_offset_text().set_fontsize(15)
    ax7.set_title(label=r'(g)Ada-methods in Breakout', fontsize=15, y=-0.4)
    ax7.grid()
    ax7.legend(fontsize=12, loc="upper left", handlelength=1)

    print("asterix_comparsion")
    for i in range(len(asterix_comparsion)):
        ys = []
        for result in asterix_comparsion[i]:
            ys.append(result[y_label].to_numpy())
        ys = np.array(ys)
        x_mean = asterix_comparsion[i][0][x_label].to_numpy()
        y_mean = np.mean(ys, axis=0)
        y_ci = np.std(ys, axis=0, ddof=0) / math.sqrt(len(ys))
        ax8.plot(x_mean, y_mean, linewidth=1.0, label=label_operation[i], color=color_operation[i])
        ax8.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, alpha=0.5)
        print(y_mean[-1])
    ax8.set_ylabel("Average score per episode", fontsize=15)
    ax8.set_xlabel("Frames", fontsize=15)
    ax8.set_xlim(0, 3e6)
    ax8.set_ylim(0, 20)
    ax8.tick_params(labelsize=15)
    xx = MultipleLocator(1e6)
    ax8.xaxis.set_major_locator(xx)
    yy = MultipleLocator(10)
    ax8.yaxis.set_major_locator(yy)
    ax8.yaxis.get_major_formatter().set_powerlimits((1, 1))
    ax8.yaxis.get_offset_text().set_fontsize(15)
    ax8.xaxis.set_major_locator(xx)
    ax8.xaxis.get_major_formatter().set_powerlimits((6, 6))
    ax8.xaxis.get_offset_text().set_fontsize(15)
    ax8.set_title(label=r'(h)Ada-methods in Asterix', fontsize=15, y=-0.4)
    ax8.grid()
    ax8.legend(fontsize=12, loc="upper left", handlelength=1)

    plt.savefig("./SuppleDeepQ.png", dpi=600, bbox_inches='tight', format='png')
    plt.show()



if __name__ == "__main__":
    x_label = 'Step'
    y_label = 'Average Return'
    Breakout_Order_ada = get_date(env_name='minatar_Breakout', x_label=x_label, y_label=y_label,
                                indexList=[1, 2, 16, 17, 18, 19],
                                runs=20)
    Asterix_Order_ada = get_date(env_name='minatar_Asterix', x_label=x_label, y_label=y_label,
                                  indexList=[1, 2, 16, 17, 18, 19],
                                  runs=20)
    copter_Order = get_date(env_name='copter', x_label=x_label, y_label=y_label,
                                 indexList=[6, 11, 12, 13, 17],
                                 runs=20)
    Breakout_Order = get_date(env_name='minatar_Breakout', x_label=x_label, y_label=y_label,
                                   indexList=[6, 11, 12, 13, 17],
                                   runs=20)
    asterix_Order = get_date(env_name='minatar_Asterix', x_label=x_label, y_label=y_label,
                                  indexList=[6, 11, 12, 13, 17],
                                  runs=20)
    copter_comparsion = get_date(env_name='copter', x_label=x_label, y_label=y_label,
                            indexList=[20, 21, 17],
                            runs=20)
    Breakout_comparsion = get_date(env_name='minatar_Breakout', x_label=x_label, y_label=y_label,
                              indexList=[20, 21, 17],
                              runs=20)
    asterix_comparsion = get_date(env_name='minatar_Asterix', x_label=x_label, y_label=y_label,
                             indexList=[20, 21, 17],
                             runs=20)

    figure(Breakout_Order_ada, Asterix_Order_ada, copter_Order, Breakout_Order, asterix_Order, copter_comparsion, Breakout_comparsion, asterix_comparsion)

