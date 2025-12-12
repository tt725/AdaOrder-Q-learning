import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import ast


def figure(MDP_o, gridworld_o):
    fig = plt.figure(figsize=(6.5, 3.5))

    # [左, 下, 宽, 高] 规定的矩形区域 （全部是0~1之间的数，表示比例）
    rect1 = [0.1, 0.26, 0.36, 0.66]
    rect2 = [0.6, 0.26, 0.36, 0.66]
    ax1 = plt.axes(rect1)
    ax2 = plt.axes(rect2)

    label_o = [r'AdaAQ(8)',
             r'AdaMQ(8)',
             r'AdaOQ(8)']


    x_value = [i for i in range(len(MDP_o[0]))]
    ax1.plot(x_value, MDP_o[0], linewidth=3.0, label=label_o[0], color='green')
    ax1.plot(x_value, MDP_o[1], linewidth=3.0, label=label_o[1], color='c')
    ax1.plot(x_value, MDP_o[2], linewidth=3.0, label=label_o[2], color='red')
    ax1.set_ylabel(r'$\Pr$[left$|$state=A]', fontsize=15)
    ax1.set_xlabel(r'Number of episodes', fontsize=15)
    ax1.set_xlim(0, len(MDP_o[0]))
    ax1.set_ylim(0, 0.6)
    ax1.tick_params(labelsize=15)
    xx = MultipleLocator(200)
    ax1.xaxis.set_major_locator(xx)
    yy = MultipleLocator(0.3)
    ax1.yaxis.set_major_locator(yy)
    ax1.yaxis.get_major_formatter().set_powerlimits((0, 1))
    ax1.yaxis.get_offset_text().set_fontsize(15)
    ax1.grid()
    ax1.legend(fontsize=12, loc="upper left", handlelength=1)
    ax1.set_title(label=r'(a)Ada-methods in MDP', fontsize=15, y=-0.4)

    x_value = [i for i in range(len(gridworld_o[0]))]
    ax2.plot(x_value, gridworld_o[0], linewidth=3.0, label=label_o[0], color='green')
    ax2.plot(x_value, gridworld_o[1], linewidth=3.0, label=label_o[1], color='c')
    ax2.plot(x_value, gridworld_o[2], linewidth=3.0, label=label_o[2], color='red')
    ax2.set_ylabel(r'Mean reward', fontsize=15)
    ax2.set_xlabel(r'Number of actions (x200)', fontsize=15)
    ax2.set_xlim(0, len(gridworld_o[0]))
    ax2.set_ylim(-0.9, 0)
    ax2.tick_params(labelsize=15)
    xx = MultipleLocator(40)
    ax2.xaxis.set_major_locator(xx)
    yy = MultipleLocator(0.4)
    ax2.yaxis.set_major_locator(yy)
    ax2.yaxis.get_major_formatter().set_powerlimits((-1, -1))
    ax2.yaxis.get_offset_text().set_fontsize(15)
    ax2.grid()
    ax2.legend(fontsize=12, loc="upper left", handlelength=1)
    ax2.set_title(label=r'(b)Ada-methods in Gridworld', fontsize=15, y=-0.4)

    plt.savefig("./SuppleOperationQ.png", dpi=600, bbox_inches='tight', format='png')
    plt.show()
    

def get_P_value(dir):
    log = open(dir, 'r').readlines()
    A_left_P = log[-1][:]
    A_left_P = ast.literal_eval(A_left_P)
    return A_left_P


def get_R_value(dir):
    log = open(dir, 'r').readlines()
    reward = log[-1][:]
    reward = ast.literal_eval(reward)
    return reward


if __name__ == "__main__":
    Q1_learning_dir = '../AdaptOrderQ/MDP/Result/log.AdaAverageQ(8,0.1) 2024.01.15.10.57.38'
    y1_value = get_P_value(
        dir=Q1_learning_dir)
    Q2_learning_dir = '../AdaptOrderQ/MDP/Result/log.AdaMaxminQ(8,0.1) 2022.11.27.01.32.01'
    y2_value = get_P_value(
        dir=Q2_learning_dir)
    Q3_learning_dir = '../AdaptOrderQ/MDP/Result/log.AdaOrderQ(8,0.1) 2022.11.25.21.37.48'
    y3_value = get_P_value(
        dir=Q3_learning_dir)
    MDP_o = [y1_value, y2_value, y3_value]

    Q1_learning_dir = '../AdaptOrderQ/Gridworld/Result/log.AdaAverageQ(8 0.1) 2024.01.07.16.23.32'
    y1_value = get_R_value(
        dir=Q1_learning_dir)
    Q2_learning_dir = '../AdaptOrderQ/Gridworld/Result/log.AdaMaxminQ(8 0.1) 2024.01.07.16.23.37'
    y2_value = get_R_value(
        dir=Q2_learning_dir)
    Q3_learning_dir = '../AdaptOrderQ/Gridworld/Result/log.AdaOrderQ(8 0.1) 2024.01.02.11.22.35'
    y3_value = get_R_value(
        dir=Q3_learning_dir)
    gridworld_o = [y1_value, y2_value, y3_value]

    figure(MDP_o, gridworld_o)