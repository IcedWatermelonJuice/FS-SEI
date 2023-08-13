import pandas as pd
import matplotlib.pyplot as plt
import os

# 设置全局字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 读取数据
def get_boxplot_data(encoder_list=["CVCNN"], freeze_list=[True, False], pretrain_list=[True, False],
                     shot_list=[5, 10, 15, 20, 25, 30], iteration_list=[100]):
    data = []
    for iteration in iteration_list:
        for encoder in encoder_list:
            for is_freezed in freeze_list:
                for is_pretrained in pretrain_list:
                    freeze = 'freezed' if is_freezed else 'unfreezed'
                    pretrain = 'pretrained' if is_pretrained else 'untrained'
                    y_list = []
                    x_list = []
                    title = f"{encoder}_{freeze}_{pretrain}_{iteration}iteration"
                    for shot in shot_list:
                        filename = f"../test_result/{encoder}_{freeze}_{pretrain}_{shot}shot_{iteration}iteration.xlsx"
                        if os.path.exists(filename):
                            y_list.append(pd.read_excel(filename)[0])
                            x_list.append(str(shot))
                    if x_list:
                        data.append([x_list, y_list, title])
    return data

# 绘制箱型图
def draw_boxplot(x_list, y_list, x_label="Shot", y_label="Accuracy (%)", title=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot(y_list, labels=x_list)
    ax.yaxis.grid(True)  # 在y轴上添加网格线
    ax.set_xlabel(x_label)  # 设置x轴名称
    ax.set_ylabel(y_label)  # 设置y轴名称
    ax.set_ylim(0, 100)  # 设置y轴的范围
    plt.savefig(f"{title}.png")
    # if title:
    #     fig.canvas.manager.set_window_title(title)


if __name__ == "__main__":
    data = get_boxplot_data()
    for d in data:
        draw_boxplot(d[0], d[1], title=d[2])
    # plt.show()
