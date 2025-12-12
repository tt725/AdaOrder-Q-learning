import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import ast


def figure(MDP_m, MDP_M, MDP_c, MDP_b, gridworld_m, gridworld_M, gridworld_c, gridworld_b):
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
    ax5 = plt.axes(rect2)
    ax2 = plt.axes(rect3)
    ax6 = plt.axes(rect4)
    ax3 = plt.axes(rect5)
    ax7 = plt.axes(rect6)
    ax4 = plt.axes(rect7)
    ax8 = plt.axes(rect8)

    label_m = [r'Q',
               r'OQ($4,1$)',
               r'OQ($4,2$)',
               r'OQ($4,4$)']
    x_value = [i for i in range(len(MDP_m[0]))]
    ax1.plot(x_value, MDP_m[0], linewidth=3.0, label=label_m[0], color='blue')
    ax1.plot(x_value, MDP_m[1], linewidth=3.0, label=label_m[1], color='green')
    ax1.plot(x_value, MDP_m[2], linewidth=3.0, label=label_m[2], color='c')
    ax1.plot(x_value, MDP_m[3], linewidth=3.0, label=label_m[3], color='m')
    ax1.set_ylabel(r'$\Pr$[left$|$state=A]', fontsize=15)
    ax1.set_xlabel(r'Number of episodes', fontsize=15)
    ax1.set_xlim(0, len(MDP_m[0]))
    ax1.set_ylim(0, 0.7)
    ax1.tick_params(labelsize=15)
    xx = MultipleLocator(200)
    ax1.xaxis.set_major_locator(xx)
    yy = MultipleLocator(0.3)
    ax1.yaxis.set_major_locator(yy)
    ax1.yaxis.get_major_formatter().set_powerlimits((0, 1))
    ax1.yaxis.get_offset_text().set_fontsize(15)
    ax1.grid()
    ax1.legend(fontsize=12, loc="upper left", handlelength=1)
    ax1.set_title(label=r'(a)Different $m$ in MDP-O', fontsize=15, y=-0.4)

    label_M = [r'DQ',
               r'OQ($4,1$)',
               r'OQ($8,1$)',
               r'OQ($16,1$)']
    x_value = [i for i in range(len(MDP_M[0]))]
    ax2.plot(x_value, MDP_M[0], linewidth=3.0, label=label_M[0], color='black')
    ax2.plot(x_value, MDP_M[1], linewidth=3.0, label=label_M[1], color='green')
    ax2.plot(x_value, MDP_M[2], linewidth=3.0, label=label_M[2], color='c')
    ax2.plot(x_value, MDP_M[3], linewidth=3.0, label=label_M[3], color='m')
    ax2.set_ylabel(r'$\Pr$[left$|$state=A]', fontsize=15)
    ax2.set_xlabel(r'Number of episodes', fontsize=15)
    ax2.set_xlim(0, len(MDP_M[0]))
    ax2.set_ylim(0, 0.6)
    ax2.tick_params(labelsize=15)
    xx = MultipleLocator(200)
    ax2.xaxis.set_major_locator(xx)
    yy = MultipleLocator(0.3)
    ax2.yaxis.set_major_locator(yy)
    ax2.yaxis.get_major_formatter().set_powerlimits((0, 1))
    ax2.yaxis.get_offset_text().set_fontsize(15)
    ax2.grid()
    ax2.legend(fontsize=12, loc="upper left", handlelength=1)
    ax2.set_title(label=r'(c)Different $M$ in MDP-U', fontsize=15, y=-0.4)

    label_c = [r'Q',
               r'DQ',
               r'AdaOQ($4,0.1$)',
               r'AdaOQ($8,0.1$)',
               r'AdaOQ($16,0.1$)']

    x_value = [i for i in range(len(MDP_c[0]))]
    ax3.plot(x_value, MDP_c[0], linewidth=3.0, label=label_c[0], color='blue')
    ax3.plot(x_value, MDP_c[1], linewidth=3.0, label=label_c[1], color='black')
    ax3.plot(x_value, MDP_c[2], linewidth=3.0, label=label_c[2], color='green')
    ax3.plot(x_value, MDP_c[3], linewidth=3.0, label=label_c[3], color='c')
    ax3.plot(x_value, MDP_c[4], linewidth=3.0, label=label_c[4], color='m')
    ax3.set_ylabel(r'$\Pr$[left$|$state=A]', fontsize=15)
    ax3.set_xlabel(r'Number of episodes', fontsize=15)
    ax3.set_xlim(0, len(MDP_c[0]))
    ax3.set_ylim(0, 0.6)
    ax3.tick_params(labelsize=15)
    xx = MultipleLocator(200)
    ax3.xaxis.set_major_locator(xx)
    yy = MultipleLocator(0.3)
    ax3.yaxis.set_major_locator(yy)
    ax3.yaxis.get_major_formatter().set_powerlimits((0, 1))
    ax3.yaxis.get_offset_text().set_fontsize(15)
    ax3.grid()
    ax3.legend(fontsize=12, loc="upper left", handlelength=1)
    ax3.set_title(label=r'(e)Different $C$ in MDP', fontsize=15, y=-0.4)

    label_b = [r'Q',
               r'DQ',
               r'AdaOQ($8,0.05$)',
               r'AdaOQ($8,0.1$)',
               r'AdaOQ($8,0.2$)']

    x_value = [i for i in range(len(MDP_b[0]))]
    ax4.plot(x_value, MDP_b[0], linewidth=3.0, label=label_b[0], color='blue')
    ax4.plot(x_value, MDP_b[1], linewidth=3.0, label=label_b[1], color='black')
    ax4.plot(x_value, MDP_b[2], linewidth=3.0, label=label_b[2], color='green')
    ax4.plot(x_value, MDP_b[3], linewidth=3.0, label=label_b[3], color='c')
    ax4.plot(x_value, MDP_b[4], linewidth=3.0, label=label_b[4], color='m')
    ax4.set_ylabel(r'$\Pr$[left$|$state=A]', fontsize=15)
    ax4.set_xlabel(r'Number of episodes', fontsize=15)
    ax4.set_xlim(0, len(MDP_b[0]))
    ax4.set_ylim(0, 0.6)
    ax4.tick_params(labelsize=15)
    xx = MultipleLocator(200)
    ax4.xaxis.set_major_locator(xx)
    yy = MultipleLocator(0.3)
    ax4.yaxis.set_major_locator(yy)
    ax4.yaxis.get_major_formatter().set_powerlimits((0, 1))
    ax4.yaxis.get_offset_text().set_fontsize(15)
    ax4.grid()
    ax4.legend(fontsize=12, loc="upper left", handlelength=1)
    ax4.set_title(label=r'(g)Different $\beta$ in MDP', fontsize=15, y=-0.4)

    x_value = [i for i in range(len(gridworld_m[0]))]
    ax5.plot(x_value, gridworld_m[0], linewidth=3.0, label=label_m[0], color='blue')
    ax5.plot(x_value, gridworld_m[1], linewidth=3.0, label=label_m[1], color='green')
    ax5.plot(x_value, gridworld_m[2], linewidth=3.0, label=label_m[2], color='c')
    ax5.plot(x_value, gridworld_m[3], linewidth=3.0, label=label_m[3], color='m')
    ax5.set_ylabel(r'Mean reward', fontsize=15)
    ax5.set_xlabel(r'Number of actions (x200)', fontsize=15)
    ax5.set_xlim(0, len(gridworld_m[0]))
    ax5.set_ylim(-0.8, 0.2)
    ax5.tick_params(labelsize=15)
    xx = MultipleLocator(40)
    ax5.xaxis.set_major_locator(xx)
    yy = MultipleLocator(0.4)
    ax5.yaxis.set_major_locator(yy)
    ax5.yaxis.get_major_formatter().set_powerlimits((-1, -1))
    ax5.yaxis.get_offset_text().set_fontsize(15)
    ax5.grid()
    ax5.legend(fontsize=12, loc="upper left", handlelength=1)
    ax5.set_title(label=r'(b)Different $m$ in Gridworld-O', fontsize=15, y=-0.4)

    x_value = [i for i in range(len(gridworld_M[0]))]
    ax6.plot(x_value, gridworld_M[0], linewidth=3.0, label=label_M[0], color='black')
    ax6.plot(x_value, gridworld_M[1], linewidth=3.0, label=label_M[1], color='green')
    ax6.plot(x_value, gridworld_M[2], linewidth=3.0, label=label_M[2], color='c')
    ax6.plot(x_value, gridworld_M[3], linewidth=3.0, label=label_M[3], color='m')
    ax6.set_ylabel(r'Mean reward', fontsize=15)
    ax6.set_xlabel(r'Number of actions (x200)', fontsize=15)
    ax6.set_xlim(0, len(gridworld_M[0]))
    ax6.set_ylim(-0.8, 0.2)
    ax6.tick_params(labelsize=15)
    xx = MultipleLocator(40)
    ax6.xaxis.set_major_locator(xx)
    yy = MultipleLocator(0.4)
    ax6.yaxis.set_major_locator(yy)
    ax6.yaxis.get_major_formatter().set_powerlimits((-1, -1))
    ax6.yaxis.get_offset_text().set_fontsize(15)
    ax6.grid()
    ax6.legend(fontsize=12, loc="upper left", handlelength=1)
    ax6.set_title(label=r'(d)Different $M$ in Gridworld-U', fontsize=15, y=-0.4)

    x_value = [i for i in range(len(gridworld_c[0]))]
    ax7.plot(x_value, gridworld_c[0], linewidth=3.0, label=label_c[0], color='blue')
    ax7.plot(x_value, gridworld_c[1], linewidth=3.0, label=label_c[1], color='black')
    ax7.plot(x_value, gridworld_c[2], linewidth=3.0, label=label_c[2], color='green')
    ax7.plot(x_value, gridworld_c[3], linewidth=3.0, label=label_c[3], color='c')
    ax7.plot(x_value, gridworld_c[4], linewidth=3.0, label=label_c[4], color='m')
    ax7.set_ylabel(r'Mean reward', fontsize=15)
    ax7.set_xlabel(r'Number of actions (x200)', fontsize=15)
    ax7.set_xlim(0, len(gridworld_b[0]))
    ax7.set_ylim(-1.0, 0)
    ax7.tick_params(labelsize=15)
    xx = MultipleLocator(40)
    ax7.xaxis.set_major_locator(xx)
    yy = MultipleLocator(0.4)
    ax7.yaxis.set_major_locator(yy)
    ax7.yaxis.get_major_formatter().set_powerlimits((-1, -1))
    ax7.yaxis.get_offset_text().set_fontsize(15)
    ax7.grid()
    ax7.legend(fontsize=12, loc="upper left", handlelength=1)
    ax7.set_title(label=r'(f)Different $C$ in Gridworld', fontsize=15, y=-0.4)

    x_value = [i for i in range(len(gridworld_b[0]))]
    ax8.plot(x_value, gridworld_b[0], linewidth=3.0, label=label_b[0], color='blue')
    ax8.plot(x_value, gridworld_b[1], linewidth=3.0, label=label_b[1], color='black')
    ax8.plot(x_value, gridworld_b[2], linewidth=3.0, label=label_b[2], color='green')
    ax8.plot(x_value, gridworld_b[3], linewidth=3.0, label=label_b[3], color='c')
    ax8.plot(x_value, gridworld_b[4], linewidth=3.0, label=label_b[4], color='m')
    ax8.set_ylabel(r'Mean reward', fontsize=15)
    ax8.set_xlabel(r'Number of actions (x200)', fontsize=15)
    ax8.set_xlim(0, len(gridworld_b[0]))
    ax8.set_ylim(-1.0, 0)
    ax8.tick_params(labelsize=15)
    xx = MultipleLocator(40)
    ax8.xaxis.set_major_locator(xx)
    yy = MultipleLocator(0.4)
    ax8.yaxis.set_major_locator(yy)
    ax8.yaxis.get_major_formatter().set_powerlimits((-1, -1))
    ax8.yaxis.get_offset_text().set_fontsize(15)
    ax8.grid()
    ax8.legend(fontsize=12, loc="upper left", handlelength=1)
    ax8.set_title(label=r'(h)Different $\beta$ in Gridworld', fontsize=15, y=-0.4)

    plt.savefig("./SuppleTableQ.png", dpi=600, bbox_inches='tight', format='png')
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
    Q1_learning_dir = '../OrderQ/MDP/HelpfulQ/Result/log.Q 2022.11.16.21.30.24'
    y1_value = get_P_value(
        dir=Q1_learning_dir)
    Q2_learning_dir = '../OrderQ/MDP/HelpfulQ/Result/log.OrderQ(4,1) 2022.11.16.21.30.05'
    y2_value = get_P_value(
        dir=Q2_learning_dir)
    Q3_learning_dir = '../OrderQ/MDP/HelpfulQ/Result/log.OrderQ(4,2) 2022.11.16.21.30.08'
    y3_value = get_P_value(
        dir=Q3_learning_dir)
    Q4_learning_dir = '../OrderQ/MDP/HelpfulQ/Result/log.OrderQ(4,4) 2022.11.16.21.30.11'
    y4_value = get_P_value(
        dir=Q4_learning_dir)
    MDP_m = [y1_value, y2_value, y3_value, y4_value]

    Q1_learning_dir = '../OrderQ/MDP/HurtfulQ/Result/log.DoubleQ 2022.11.16.21.53.14'
    y1_value = get_P_value(
        dir=Q1_learning_dir)
    Q2_learning_dir = '../OrderQ/MDP/HurtfulQ/Result/log.OrderQ(4,1) 2022.11.16.21.53.37'
    y2_value = get_P_value(
        dir=Q2_learning_dir)
    Q3_learning_dir = '../OrderQ/MDP/HurtfulQ/Result/log.OrderQ(8,1) 2022.11.16.21.53.40'
    y3_value = get_P_value(
        dir=Q3_learning_dir)
    Q4_learning_dir = '../OrderQ/MDP/HurtfulQ/Result/log.OrderQ(16,1) 2022.11.16.21.53.42'
    y4_value = get_P_value(
        dir=Q4_learning_dir)
    MDP_M = [y1_value, y2_value, y3_value, y4_value]

    Q1_learning_dir = '../AdaptOrderQ/MDP/Result/log.Q 2022.11.25.21.39.10'
    y1_value = get_P_value(
        dir=Q1_learning_dir)
    Q2_learning_dir = '../AdaptOrderQ/MDP/Result/log.DoubleQ 2022.11.25.21.39.02'
    y2_value = get_P_value(
        dir=Q2_learning_dir)
    Q3_learning_dir = '../AdaptOrderQ/MDP/Result/log.AdaOrderQ(4,0.1) 2022.11.25.21.37.45'
    y3_value = get_P_value(
        dir=Q3_learning_dir)
    Q4_learning_dir = '../AdaptOrderQ/MDP/Result/log.AdaOrderQ(8,0.1) 2022.11.25.21.37.48'
    y4_value = get_P_value(
        dir=Q4_learning_dir)
    Q5_learning_dir = '../AdaptOrderQ/MDP/Result/log.AdaOrderQ(16,0.1) 2022.11.25.21.38.54'
    y5_value = get_P_value(
        dir=Q5_learning_dir)
    MDP_c = [y1_value, y2_value, y3_value, y4_value, y5_value]

    Q1_learning_dir = '../AdaptOrderQ/MDP/Result/log.Q 2022.11.25.21.39.10'
    y1_value = get_P_value(
        dir=Q1_learning_dir)
    Q2_learning_dir = '../AdaptOrderQ/MDP/Result/log.DoubleQ 2022.11.25.21.39.02'
    y2_value = get_P_value(
        dir=Q2_learning_dir)
    Q3_learning_dir = '../AdaptOrderQ/MDP/Result/log.AdaOrderQ(8,0.05) 2022.11.25.21.38.42'
    y3_value = get_P_value(
        dir=Q3_learning_dir)
    Q4_learning_dir = '../AdaptOrderQ/MDP/Result/log.AdaOrderQ(8,0.1) 2022.11.25.21.37.48'
    y4_value = get_P_value(
        dir=Q4_learning_dir)
    Q5_learning_dir = '../AdaptOrderQ/MDP/Result/log.AdaOrderQ(8,0.2) 2022.11.25.21.37.51'
    y5_value = get_P_value(
        dir=Q5_learning_dir)
    MDP_b = [y1_value, y2_value, y3_value, y4_value, y5_value]

    print("MDP:")
    for i in MDP_b:
        print(i[-1])

    Q1_learning_dir = '../OrderQ/Gridworld/Helpful/Result/log.Q 2024.01.02.16.06.50'
    y1_value = get_R_value(
        dir=Q1_learning_dir)
    Q2_learning_dir = '../OrderQ/Gridworld/Helpful/Result/log.OrderQ(4 1) 2024.01.02.16.06.41'
    y2_value = get_R_value(
        dir=Q2_learning_dir)
    Q3_learning_dir = '../OrderQ/Gridworld/Helpful/Result/log.OrderQ(4 2) 2024.01.02.16.06.43'
    y3_value = get_R_value(
        dir=Q3_learning_dir)
    Q4_learning_dir = '../OrderQ/Gridworld/Helpful/Result/log.OrderQ(4 4) 2024.01.02.16.06.46'
    y4_value = get_R_value(
        dir=Q4_learning_dir)
    gridworld_m = [y1_value, y2_value, y3_value, y4_value]

    Q1_learning_dir = '../OrderQ/Gridworld/Hurtful/Result/log.DoubleQ 2024.01.02.16.43.59'
    y1_value = get_R_value(
        dir=Q1_learning_dir)
    Q2_learning_dir = '../OrderQ/Gridworld/Hurtful/Result/log.OrderQ(4 1) 2022.11.17.21.34.26'
    y2_value = get_R_value(
        dir=Q2_learning_dir)
    Q3_learning_dir = '../OrderQ/Gridworld/Hurtful/Result/log.OrderQ(8 1) 2024.01.02.16.44.04'
    y3_value = get_R_value(
        dir=Q3_learning_dir)
    Q4_learning_dir = '../OrderQ/Gridworld/Hurtful/Result/log.OrderQ(16 1) 2024.01.02.16.44.06'
    y4_value = get_R_value(
        dir=Q4_learning_dir)
    gridworld_M = [y1_value, y2_value, y3_value, y4_value]

    Q1_learning_dir = '../AdaptOrderQ/Gridworld/Result/log.Q 2024.01.02.13.53.38'
    y1_value = get_R_value(
        dir=Q1_learning_dir)
    Q2_learning_dir = '../AdaptOrderQ/Gridworld/Result/log.DoubleQ 2024.01.02.13.53.43'
    y2_value = get_R_value(
        dir=Q2_learning_dir)
    Q3_learning_dir = '../AdaptOrderQ/Gridworld/Result/log.AdaOrderQ(4 0.1) 2024.01.02.13.53.56'
    y3_value = get_R_value(
        dir=Q3_learning_dir)
    Q4_learning_dir = '../AdaptOrderQ/Gridworld/Result/log.AdaOrderQ(8 0.1) 2024.01.02.11.22.35'
    y4_value = get_R_value(
        dir=Q4_learning_dir)
    Q5_learning_dir = '../AdaptOrderQ/Gridworld/Result/log.AdaOrderQ(16 0.1) 2024.01.02.13.54.22'
    y5_value = get_R_value(
        dir=Q5_learning_dir)
    gridworld_c = [y1_value, y2_value, y3_value, y4_value, y5_value]

    print("reward:")
    for i in gridworld_c:
        print(i[-1])

    Q1_learning_dir = '../AdaptOrderQ/Gridworld/Result/log.Q 2024.01.02.13.53.38'
    y1_value = get_R_value(
        dir=Q1_learning_dir)
    Q2_learning_dir = '../AdaptOrderQ/Gridworld/Result/log.DoubleQ 2024.01.02.13.53.43'
    y2_value = get_R_value(
        dir=Q2_learning_dir)
    Q3_learning_dir = '../AdaptOrderQ/Gridworld/Result/log.AdaOrderQ(8 0.05) 2024.01.02.13.54.12'
    y3_value = get_R_value(
        dir=Q3_learning_dir)
    Q4_learning_dir = '../AdaptOrderQ/Gridworld/Result/log.AdaOrderQ(8 0.1) 2024.01.02.11.22.35'
    y4_value = get_R_value(
        dir=Q4_learning_dir)
    Q5_learning_dir = '../AdaptOrderQ/Gridworld/Result/log.AdaOrderQ(8 0.2) 2024.01.02.13.54.05'
    y5_value = get_R_value(
        dir=Q5_learning_dir)
    gridworld_b = [y1_value, y2_value, y3_value, y4_value, y5_value]

    figure(MDP_m, MDP_M, MDP_c, MDP_b, gridworld_m, gridworld_M, gridworld_c, gridworld_b)

