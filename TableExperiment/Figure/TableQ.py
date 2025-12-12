import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import ast


def figure(multiarm_m, multiarm_M, multiarm_c, multiarm_b, multiarm_sota, MDP_sota, gridworld_sota, multiarm_operation):
    fig = plt.figure(figsize=(13, 7))
    # [左, 下, 宽, 高] 规定的矩形区域 （全部是0~1之间的数，表示比例）
    rect1 = [0.05, 0.65, 0.18, 0.34]
    rect2 = [0.3, 0.65, 0.18, 0.34]
    rect3 = [0.55, 0.65, 0.18, 0.34]
    rect4 = [0.8, 0.65, 0.18, 0.34]
    rect5 = [0.05, 0.13, 0.18, 0.34]
    rect6 = [0.3, 0.13, 0.18, 0.34]
    rect7 = [0.55, 0.13, 0.18, 0.34]
    rect8 = [0.8, 0.13, 0.18, 0.34]
    ax1 = plt.axes(rect1)
    ax2 = plt.axes(rect2)
    ax3 = plt.axes(rect3)
    ax4 = plt.axes(rect4)
    ax5 = plt.axes(rect5)
    ax6 = plt.axes(rect6)
    ax7 = plt.axes(rect7)
    ax8 = plt.axes(rect8)

    label_m = [r'Q',
               r'OQ($4,1$)',
               r'OQ($4,2$)',
               r'OQ($4,4$)']
    x_value = [i for i in range(len(multiarm_m[0]))]
    ax1.plot(x_value, multiarm_m[0], linewidth=3.0, label=label_m[0], color='blue')
    ax1.plot(x_value, multiarm_m[1], linewidth=3.0, label=label_m[1], color='green')
    ax1.plot(x_value, multiarm_m[2], linewidth=3.0, label=label_m[2], color='c')
    ax1.plot(x_value, multiarm_m[3], linewidth=3.0, label=label_m[3], color='m')
    ax1.set_ylabel(r'Maximum Q-value', fontsize=15)
    ax1.set_xlabel(r'Number of actions (x100)', fontsize=15)
    ax1.set_xlim(0, len(multiarm_m[0]))
    ax1.set_ylim(-1, 11.0)
    ax1.tick_params(labelsize=15)
    xx = MultipleLocator(40)
    ax1.xaxis.set_major_locator(xx)
    yy = MultipleLocator(5)
    ax1.yaxis.set_major_locator(yy)
    ax1.yaxis.get_offset_text().set_fontsize(15)
    ax1.grid()
    ax1.legend(fontsize=12, loc="upper left", handlelength=1)
    ax1.set_title(label=r'(a)OQ with different $m$', fontsize=15, y=-0.4)

    label_M = [r'DQ',
               r'OQ($4,1$)',
               r'OQ($8,1$)',
               r'OQ($16,1$)']
    x_value = [i for i in range(len(multiarm_M[0]))]
    ax2.plot(x_value, multiarm_M[0], linewidth=3.0, label=label_M[0], color='black')
    ax2.plot(x_value, multiarm_M[1], linewidth=3.0, label=label_M[1], color='green')
    ax2.plot(x_value, multiarm_M[2], linewidth=3.0, label=label_M[2], color='c')
    ax2.plot(x_value, multiarm_M[3], linewidth=3.0, label=label_M[3], color='m')
    ax2.set_ylabel(r'Maximum Q-value', fontsize=15)
    ax2.set_xlabel(r'Number of actions (x100)', fontsize=15)
    ax2.set_xlim(0, len(multiarm_m[0]))
    ax2.set_ylim(-2.5, 0.5)
    ax2.tick_params(labelsize=15)
    xx = MultipleLocator(40)
    ax2.xaxis.set_major_locator(xx)
    yy = MultipleLocator(1)
    ax2.yaxis.set_major_locator(yy)
    ax2.yaxis.get_offset_text().set_fontsize(15)
    ax2.grid()
    ax2.legend(fontsize=12, loc="upper left", handlelength=1)
    ax2.set_title(label=r'(b)OQ with different $M$', fontsize=15, y=-0.4)

    label_c = [r'Q',
             r'DQ',
             r'AdaOQ($4,0.1$)',
             r'AdaOQ($8,0.1$)',
             r'AdaOQ($16,0.1$)']
    x_value = [i for i in range(len(multiarm_c[0]))]
    ax3.plot(x_value, multiarm_c[0], linewidth=3.0, label=label_c[0], color='blue')
    ax3.plot(x_value, multiarm_c[1], linewidth=3.0, label=label_c[1], color='black')
    ax3.plot(x_value, multiarm_c[2], linewidth=3.0, label=label_c[2], color='green')
    ax3.plot(x_value, multiarm_c[3], linewidth=3.0, label=label_c[3], color='c')
    ax3.plot(x_value, multiarm_c[4], linewidth=3.0, label=label_c[4], color='m')
    ax3.set_ylabel(r'Maximum Q-value', fontsize=15)
    ax3.set_xlabel(r'Number of actions (x100)', fontsize=15)
    ax3.set_xlim(0, len(multiarm_c[0]))
    ax3.set_ylim(-3.0, 3.5)
    ax3.tick_params(labelsize=15)
    xx = MultipleLocator(40)
    ax3.xaxis.set_major_locator(xx)
    yy = MultipleLocator(3)
    ax3.yaxis.set_major_locator(yy)
    ax3.yaxis.get_offset_text().set_fontsize(15)
    ax3.grid()
    ax3.legend(fontsize=12, loc="upper left", handlelength=1)
    ax3.set_title(label=r'(c)AdaOQ with different $C$', fontsize=15, y=-0.4)

    label_b = [r'Q',
               r'DQ',
               r'AdaOQ($8,0.05$)',
               r'AdaOQ($8,0.1$)',
               r'AdaOQ($8,0.2$)']
    x_value = [i for i in range(len(multiarm_b[0]))]
    ax4.plot(x_value, multiarm_b[0], linewidth=3.0, label=label_b[0], color='blue')
    ax4.plot(x_value, multiarm_b[1], linewidth=3.0, label=label_b[1], color='black')
    ax4.plot(x_value, multiarm_b[2], linewidth=3.0, label=label_b[2], color='green')
    ax4.plot(x_value, multiarm_b[3], linewidth=3.0, label=label_b[3], color='c')
    ax4.plot(x_value, multiarm_b[4], linewidth=3.0, label=label_b[4], color='m')
    ax4.set_ylabel(r'Maximum Q-value', fontsize=15)
    ax4.set_xlabel(r'Number of actions (x100)', fontsize=15)
    ax4.set_xlim(0, len(multiarm_c[0]))
    ax4.set_ylim(-3.0, 3.5)
    ax4.tick_params(labelsize=15)
    xx = MultipleLocator(40)
    ax4.xaxis.set_major_locator(xx)
    yy = MultipleLocator(3)
    ax4.yaxis.set_major_locator(yy)
    ax4.yaxis.get_offset_text().set_fontsize(15)
    ax4.grid()
    ax4.legend(fontsize=12, loc="upper left", handlelength=1)
    ax4.set_title(label=r'(d)AdaOQ with different $\beta$', fontsize=15, y=-0.4)

    label_sota = [r'EQ',
             r"MQ",
             r'SCQ',
             r'SQ',
             r'WDQ',
             r'REDQ',
             r'EBQL',
             r"AdaEQ",
             r'AdaOQ']

    x_value = [i for i in range(len(multiarm_sota[0]))]
    ax5.plot(x_value, multiarm_sota[0], linewidth=3.0, label=label_sota[0], color='blue')
    ax5.plot(x_value, multiarm_sota[1], linewidth=3.0, label=label_sota[1], color='black')
    ax5.plot(x_value, multiarm_sota[2], linewidth=3.0, label=label_sota[2], color='green')
    ax5.plot(x_value, multiarm_sota[3], linewidth=3.0, label=label_sota[3], color='c')
    ax5.plot(x_value, multiarm_sota[4], linewidth=3.0, label=label_sota[4], color='m')
    ax5.plot(x_value, multiarm_sota[5], linewidth=3.0, label=label_sota[5], color='y')
    ax5.plot(x_value, multiarm_sota[6], linewidth=3.0, label=label_sota[6], color='#1f77b4')
    ax5.plot(x_value, multiarm_sota[7], linewidth=3.0, label=label_sota[7], color='#ff7f0e')
    ax5.plot(x_value, multiarm_sota[8], linewidth=3.0, label=label_sota[8], color='red')
    ax5.set_ylabel(r'Maximum Q-value', fontsize=15)
    ax5.set_xlabel(r'Number of actions (x100)', fontsize=15)
    ax5.set_xlim(0, len(multiarm_sota[0]))
    ax5.set_ylim(-2.2, 3.0)
    ax5.tick_params(labelsize=15)
    xx = MultipleLocator(40)
    ax5.xaxis.set_major_locator(xx)
    yy = MultipleLocator(2)
    ax5.yaxis.set_major_locator(yy)
    ax5.yaxis.get_major_formatter().set_powerlimits((0, 1))
    ax5.yaxis.get_offset_text().set_fontsize(15)
    ax5.grid()
    ax5.legend(fontsize=12, loc="upper left", handlelength=1)
    ax5.set_title(label=r'(e)Comparison in Multi-armed', fontsize=15, y=-0.4)

    x_value = [i for i in range(len(MDP_sota[0]))]
    ax6.plot(x_value, MDP_sota[0], linewidth=3.0, label=label_sota[0], color='blue')
    ax6.plot(x_value, MDP_sota[1], linewidth=3.0, label=label_sota[1], color='black')
    ax6.plot(x_value, MDP_sota[2], linewidth=3.0, label=label_sota[2], color='green')
    ax6.plot(x_value, MDP_sota[3], linewidth=3.0, label=label_sota[3], color='c')
    ax6.plot(x_value, MDP_sota[4], linewidth=3.0, label=label_sota[4], color='m')
    ax6.plot(x_value, MDP_sota[5], linewidth=3.0, label=label_sota[5], color='y')
    ax6.plot(x_value, MDP_sota[6], linewidth=3.0, label=label_sota[6], color='#1f77b4')
    ax6.plot(x_value, MDP_sota[7], linewidth=3.0, label=label_sota[7], color='#ff7f0e')
    ax6.plot(x_value, MDP_sota[8], linewidth=3.0, label=label_sota[8], color='red')
    ax6.set_ylabel(r'$\Pr$[left$|$state=A]', fontsize=15)
    ax6.set_xlabel(r'Number of episodes', fontsize=15)
    ax6.set_xlim(0, len(MDP_sota[0]))
    ax6.set_ylim(0, 0.6)
    ax6.tick_params(labelsize=15)
    xx = MultipleLocator(200)
    ax6.xaxis.set_major_locator(xx)
    yy = MultipleLocator(0.3)
    ax6.yaxis.set_major_locator(yy)
    ax6.yaxis.get_major_formatter().set_powerlimits((0, 1))
    ax6.yaxis.get_offset_text().set_fontsize(15)
    ax6.grid()
    ax6.legend(fontsize=12, loc="upper left", handlelength=1)
    ax6.set_title(label=r'(f)Comparison in MDP', fontsize=15, y=-0.4)

    x_value = [i for i in range(len(gridworld_sota[0]))]
    ax7.plot(x_value, gridworld_sota[0], linewidth=3.0, label=label_sota[0], color='blue')
    ax7.plot(x_value, gridworld_sota[1], linewidth=3.0, label=label_sota[1], color='black')
    ax7.plot(x_value, gridworld_sota[2], linewidth=3.0, label=label_sota[2], color='green')
    ax7.plot(x_value, gridworld_sota[3], linewidth=3.0, label=label_sota[3], color='c')
    ax7.plot(x_value, gridworld_sota[4], linewidth=3.0, label=label_sota[4], color='m')
    ax7.plot(x_value, gridworld_sota[5], linewidth=3.0, label=label_sota[5], color='y')
    ax7.plot(x_value, gridworld_sota[6], linewidth=3.0, label=label_sota[6], color='#1f77b4')
    ax7.plot(x_value, gridworld_sota[7], linewidth=3.0, label=label_sota[7], color='#ff7f0e')
    ax7.plot(x_value, gridworld_sota[8], linewidth=3.0, label=label_sota[8], color='red')
    ax7.set_ylabel(r'Mean reward', fontsize=15)
    ax7.set_xlabel(r'Number of actions (x200)', fontsize=15)
    ax7.set_xlim(0, len(gridworld_sota[0]))
    ax7.set_ylim(-0.99, -0.0)
    ax7.tick_params(labelsize=15)
    xx = MultipleLocator(40)
    ax7.xaxis.set_major_locator(xx)
    yy = MultipleLocator(0.4)
    ax7.yaxis.set_major_locator(yy)
    ax7.yaxis.get_major_formatter().set_powerlimits((-1, -1))
    ax7.yaxis.get_offset_text().set_fontsize(15)
    ax7.grid()
    ax7.legend(fontsize=12, loc="upper left", handlelength=1)
    ax7.set_title(label=r'(g)Comparison in Gridworld', fontsize=15, y=-0.4)

    label = [r'AdaAQ',
             r"AdaMQ",
             r'AdaOQ']

    x_value = [i for i in range(len(multiarm_operation[0]))]
    ax8.plot(x_value, multiarm_operation[0], linewidth=3.0, label=label[0], color='green')
    ax8.plot(x_value, multiarm_operation[1], linewidth=3.0, label=label[1], color='c')
    ax8.plot(x_value, multiarm_operation[2], linewidth=3.0, label=label[2], color='red')
    ax8.set_ylabel(r'Maximum Q-value', fontsize=15)
    ax8.set_xlabel(r'Number of actions (x100)', fontsize=15)
    ax8.set_xlim(0, len(multiarm_operation[0]))
    ax8.set_ylim(-0.99, 0.99)
    ax8.tick_params(labelsize=15)
    xx = MultipleLocator(40)
    ax8.xaxis.set_major_locator(xx)
    yy = MultipleLocator(0.5)
    ax8.yaxis.set_major_locator(yy)
    ax8.yaxis.get_major_formatter().set_powerlimits((-1, -1))
    ax8.yaxis.get_offset_text().set_fontsize(15)
    ax8.grid()
    ax8.legend(fontsize=12, loc="upper left", handlelength=1)
    ax8.set_title(label=r'(h)Three adaptive methods', fontsize=15, y=-0.4)

    plt.savefig("./TableQ.png", dpi=600, bbox_inches='tight', format='png')
    plt.show()
    


def get_Q_value(dir):
    log = open(dir, 'r').readlines()
    max_Q_S = log[-1][:]
    max_Q_S = ast.literal_eval(max_Q_S)
    return max_Q_S


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
    Q1_learning_dir = '../OrderQ/MultiArms/Result/log.Q 2022.10.25.18.00.35'
    y1_value = get_Q_value(
        dir=Q1_learning_dir)
    Q2_learning_dir = '../OrderQ/MultiArms/Result/log.OrderQ(4,1) 2022.11.19.22.28.46'
    y2_value = get_Q_value(
        dir=Q2_learning_dir)
    Q3_learning_dir = '../OrderQ/MultiArms/Result/log.OrderQ(4,2) 2022.11.19.22.28.50'
    y3_value = get_Q_value(
        dir=Q3_learning_dir)
    Q4_learning_dir = '../OrderQ/MultiArms/Result/log.OrderQ(4,4) 2022.11.19.22.28.53'
    y4_value = get_Q_value(
        dir=Q4_learning_dir)
    multiarm_m = [y1_value, y2_value, y3_value, y4_value]

    Q1_learning_dir = '../OrderQ/MultiArms/Result/log.DoubleQ 2022.10.25.17.58.27'
    y1_value = get_Q_value(
        dir=Q1_learning_dir)
    Q2_learning_dir = '../OrderQ/MultiArms/Result/log.OrderQ(4,1) 2022.11.19.22.28.46'
    y2_value = get_Q_value(
        dir=Q2_learning_dir)
    Q3_learning_dir = '../OrderQ/MultiArms/Result/log.OrderQ(8,1) 2022.11.19.22.29.36'
    y3_value = get_Q_value(
        dir=Q3_learning_dir)
    Q4_learning_dir = '../OrderQ/MultiArms/Result/log.OrderQ(16,1) 2022.11.19.22.29.45'
    y4_value = get_Q_value(
        dir=Q4_learning_dir)
    multiarm_M = [y1_value, y2_value, y3_value, y4_value]

    Q1_learning_dir = '../AdaptOrderQ/MultiArms/Result/log.Q 2022.11.10.15.59.44'
    y1_value = get_Q_value(
        dir=Q1_learning_dir)
    Q2_learning_dir = '../AdaptOrderQ/MultiArms/Result/log.DoubleQ 2022.11.10.15.59.41'
    y2_value = get_Q_value(
        dir=Q2_learning_dir)
    Q3_learning_dir = '../AdaptOrderQ/MultiArms/Result/log.AdaOrderQ(4,0.1) 2022.11.21.21.07.02'
    y3_value = get_Q_value(
        dir=Q3_learning_dir)
    Q4_learning_dir = '../AdaptOrderQ/MultiArms/Result/log.AdaOrderQ(8,0.1) 2022.11.21.21.07.05'
    y4_value = get_Q_value(
        dir=Q4_learning_dir)
    Q5_learning_dir = '../AdaptOrderQ/MultiArms/Result/log.AdaOrderQ(16,0.1) 2022.11.21.21.07.33'
    y5_value = get_Q_value(
        dir=Q5_learning_dir)
    multiarm_c = [y1_value, y2_value, y3_value, y4_value, y5_value]

    print("maximum Q value:")
    for i in multiarm_c:
        print(i[-1])

    Q1_learning_dir = '../AdaptOrderQ/MultiArms/Result/log.Q 2022.11.10.15.59.44'
    y1_value = get_Q_value(
        dir=Q1_learning_dir)
    Q2_learning_dir = '../AdaptOrderQ/MultiArms/Result/log.DoubleQ 2022.11.10.15.59.41'
    y2_value = get_Q_value(
        dir=Q2_learning_dir)
    Q3_learning_dir = '../AdaptOrderQ/MultiArms/Result/log.AdaOrderQ(8,0.05) 2022.11.21.21.07.30'
    y3_value = get_Q_value(
        dir=Q3_learning_dir)
    Q4_learning_dir = '../AdaptOrderQ/MultiArms/Result/log.AdaOrderQ(8,0.1) 2022.11.21.21.07.05'
    y4_value = get_Q_value(
        dir=Q4_learning_dir)
    Q5_learning_dir = '../AdaptOrderQ/MultiArms/Result/log.AdaOrderQ(8,0.2) 2022.11.21.21.07.08'
    y5_value = get_Q_value(
        dir=Q5_learning_dir)
    multiarm_b = [y1_value, y2_value, y3_value, y4_value, y5_value]

    Q1_learning_dir = '../AdaptOrderQ/MultiArms/Result/log.EnsembleQ(8) 2022.11.26.17.29.34'
    y1_value = get_Q_value(
        dir=Q1_learning_dir)
    Q2_learning_dir = '../AdaptOrderQ/MultiArms/Result/log.MaxminQ(8) 2022.11.26.17.29.38'
    y2_value = get_Q_value(
        dir=Q2_learning_dir)
    Q3_learning_dir = '../AdaptOrderQ/MultiArms/Result/log.SelfCorrectingQ 2022.11.26.17.29.43'
    y3_value = get_Q_value(
        dir=Q3_learning_dir)
    Q4_learning_dir = '../AdaptOrderQ/MultiArms/Result/log.SoftmaxQ 2022.11.26.17.29.46'
    y4_value = get_Q_value(
        dir=Q4_learning_dir)
    Q5_learning_dir = '../AdaptOrderQ/MultiArms/Result/log.WeightedDoubleQ(0.5) 2022.11.26.17.30.02'
    y5_value = get_Q_value(
        dir=Q5_learning_dir)
    Q6_learning_dir = '../AdaptOrderQ/MultiArms/Result/log.REDQ(8) 2023.12.15.17.03.07'
    y6_value = get_Q_value(
        dir=Q6_learning_dir)
    Q7_learning_dir = '../AdaptOrderQ/MultiArms/Result/log.EBQL(8) 2023.12.15.16.59.52'
    y7_value = get_Q_value(
        dir=Q7_learning_dir)
    Q8_learning_dir = '../AdaptOrderQ/MultiArms/Result/log.AdaEQ(8) 2023.12.18.17.05.21'
    y8_value = get_Q_value(
        dir=Q8_learning_dir)
    Q9_learning_dir = '../AdaptOrderQ/MultiArms/Result/log.AdaOrderQ(8,0.1) 2022.11.21.21.07.05'
    y9_value = get_Q_value(
        dir=Q9_learning_dir)
    multiarm_sota = [y1_value, y2_value, y3_value, y4_value, y5_value, y6_value, y7_value, y8_value, y9_value]

    print("maximum Q-value:")
    for i in multiarm_sota:
        print(i[-1])


    Q1_learning_dir = '../AdaptOrderQ/MDP/Result/log.EnsembleQ(8) 2024.01.11.17.51.35'
    y1_value = get_P_value(
        dir=Q1_learning_dir)
    Q2_learning_dir = '../AdaptOrderQ/MDP/Result/log.MaxminQ(8) 2022.11.26.19.57.26'
    y2_value = get_P_value(
        dir=Q2_learning_dir)
    Q3_learning_dir = '../AdaptOrderQ/MDP/Result/log.SelfCorrectingQ 2022.11.26.19.54.23'
    y3_value = get_P_value(
        dir=Q3_learning_dir)
    Q4_learning_dir = '../AdaptOrderQ/MDP/Result/log.SoftmaxQ 2022.11.26.19.54.29'
    y4_value = get_P_value(
        dir=Q4_learning_dir)
    Q5_learning_dir = '../AdaptOrderQ/MDP/Result/log.WeightedDoubleQ 2022.11.26.19.54.40'
    y5_value = get_P_value(
        dir=Q5_learning_dir)
    Q6_learning_dir = '../AdaptOrderQ/MDP/Result/log.REDQ(8) 2023.12.15.17.28.10'
    y6_value = get_P_value(
        dir=Q6_learning_dir)
    Q7_learning_dir = '../AdaptOrderQ/MDP/Result/log.EBQL(8) 2023.12.15.17.24.11'
    y7_value = get_P_value(
        dir=Q7_learning_dir)
    Q8_learning_dir = '../AdaptOrderQ/MDP/Result/log.AdaEQ(8) 2023.12.18.17.05.25'
    y8_value = get_P_value(
        dir=Q8_learning_dir)
    Q9_learning_dir = '../AdaptOrderQ/MDP/Result/log.AdaOrderQ(8,0.1) 2022.11.25.21.37.48'
    y9_value = get_P_value(
        dir=Q9_learning_dir)
    MDP_sota = [y1_value, y2_value, y3_value, y4_value, y5_value, y6_value, y7_value, y8_value, y9_value]

    print("mdp:")
    for i in MDP_sota:
        print(i[-1])

    Q1_learning_dir = '../AdaptOrderQ/Gridworld/Result/log.EnsembleQ(8) 2024.01.02.11.24.26'
    y1_value = get_R_value(
        dir=Q1_learning_dir)
    Q2_learning_dir = '../AdaptOrderQ/Gridworld/Result/log.MaxminQ(8) 2024.01.02.11.24.35'
    y2_value = get_R_value(
        dir=Q2_learning_dir)
    Q3_learning_dir = '../AdaptOrderQ/Gridworld/Result/log.SelfCorrectingQ 2024.01.02.11.25.06'
    y3_value = get_R_value(
        dir=Q3_learning_dir)
    Q4_learning_dir = '../AdaptOrderQ/Gridworld/Result/log.SoftmaxQ 2024.01.02.11.25.11'
    y4_value = get_R_value(
        dir=Q4_learning_dir)
    Q5_learning_dir = '../AdaptOrderQ/Gridworld/Result/log.WeightedDoubleQ 2024.01.02.11.25.32'
    y5_value = get_R_value(
        dir=Q5_learning_dir)
    Q6_learning_dir = '../AdaptOrderQ/Gridworld/Result/log.REDQ(8) 2024.01.02.11.24.40'
    y6_value = get_R_value(
        dir=Q6_learning_dir)
    Q7_learning_dir = '../AdaptOrderQ/Gridworld/Result/log.EBQL(8) 2024.01.02.13.53.19'
    y7_value = get_R_value(
        dir=Q7_learning_dir)
    Q8_learning_dir = '../AdaptOrderQ/Gridworld/Result/log.AdaEQ(8) 2024.01.02.11.24.11'
    y8_value = get_R_value(
        dir=Q8_learning_dir)
    Q9_learning_dir = '../AdaptOrderQ/Gridworld/Result/log.AdaOrderQ(8 0.1) 2024.01.02.11.22.35'
    y9_value = get_R_value(
        dir=Q9_learning_dir)
    gridworld_sota = [y1_value, y2_value, y3_value, y4_value, y5_value, y6_value, y7_value, y8_value, y9_value]

    print("reward:")
    for i in gridworld_sota:
        print(i[-1])

    Q1_learning_dir = '../AdaptOrderQ/MultiArms/Result/log.AdaAverageQ(8,0.1) 2022.11.27.01.30.55'
    y1_value = get_Q_value(
        dir=Q1_learning_dir)
    Q2_learning_dir = '../AdaptOrderQ/MultiArms/Result/log.AdaMaxminQ(8,0.1) 2022.11.27.01.30.58'
    y2_value = get_Q_value(
        dir=Q2_learning_dir)
    Q3_learning_dir = '../AdaptOrderQ/MultiArms/Result/log.AdaOrderQ(8,0.1) 2022.11.21.21.07.05'
    y3_value = get_Q_value(
        dir=Q3_learning_dir)
    multiarm_operation = [y1_value, y2_value, y3_value]

    figure(multiarm_m, multiarm_M, multiarm_c, multiarm_b, multiarm_sota, MDP_sota, gridworld_sota, multiarm_operation)

