import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 读取数据
shots=["5","10","15","20","25","30"]
data={
    "pretrain":[np.mean(pd.read_excel(f"../test_result/CVCNN_unfreezed_pretrained_{shot}shot_100iteration.xlsx")[0]) for shot in shots],
    "untrain":[np.mean(pd.read_excel(f"../test_result/CVCNN_unfreezed_untrained_{shot}shot_100iteration.xlsx")[0]) for shot in shots]
}

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(shots, data["pretrain"], "-o", label="FS-SEI based on RFD")
ax.plot(shots, data["untrain"], "-o", label="FS-SEI without pretraining")
ax.yaxis.grid(True)  # 在y轴上添加网格线
ax.set_xlabel("Shot")  # 设置x轴名称
ax.set_ylabel("Accuracy (%)")  # 设置y轴名称
ax.set_ylim(20, 100)  # 设置y轴的范围
ax.legend()

plt.savefig(f"average_comparison.png")