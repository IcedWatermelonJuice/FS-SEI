import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置全局字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'# 读取数据
def get_average_data(ablated_loss=["Origin", "CLS", "MSE", "NCE"], freeze_list=[True, False],
                     pretrain_list=[True, False], shot_list=[10, 20, 30]):
    data = []
    for loss in ablated_loss:
        for is_freezed in freeze_list:
            for is_pretrained in pretrain_list:
                freeze = 'freezed' if is_freezed else 'unfreezed'
                pretrain = 'pretrained' if is_pretrained else 'untrained'
                y_list = []
                x_list = []
                title = f"w/o {loss}" if loss != "Origin" else "RFD"
                for shot in shot_list:
                    filename = f"../test_result/{loss}_{freeze}_{pretrain}_{shot}shot.xlsx"
                    if os.path.exists(filename):
                        y_list.append(pd.read_excel(filename)[0])
                        x_list.append(str(shot))
                if x_list:
                    data.append([x_list, np.mean(y_list, axis=1), title])
    return data


# 绘制箱型图
def draw_average_line_chart(data, x_label="Shot", y_label="Accuracy (%)", title=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    for d in data:
        ax.plot(d[0], d[1], "-o", label=d[2])
    ax.yaxis.grid(True)  # 在y轴上添加网格线
    ax.set_xlabel(x_label)  # 设置x轴名称
    ax.set_ylabel(y_label)  # 设置y轴名称
    ax.set_ylim(20, 100)  # 设置y轴的范围
    ax.legend()
    plt.savefig(f"{title}.png")



if __name__ == "__main__":
    data = get_average_data()
    draw_average_line_chart(data, title="comparation_of_ablation")
