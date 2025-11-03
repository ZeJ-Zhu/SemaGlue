import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置字体大小
plt.rcParams.update({
    'font.size': 18,         # 全局字体大小
    'axes.titlesize': 26,    # 标题字体大小
    'axes.labelsize': 18,    # 轴标签字体大小
    'xtick.labelsize': 24,   # x轴刻度字体大小
    'ytick.labelsize': 24,   # y轴刻度字体大小
    'legend.fontsize': 22    # 图例字体大小
})

# 创建数据框架
data = {
    "Keypoints": [512, 1024, 2048, 4096, 8192],
    "SemaGlue_time": [60.5502, 61.4625, 84.5645, 208.3004, 714.5935],
    "SGMNet_time": [50.8007, 50.6555, 51.7695, 66.5256, 127.7146],
    "SuperGlue_time": [27.4191, 27.6679, 49.002, 169.7118, 742.368],
    "LightGlue_time": [17.3984, 17.4153, 36.2952, 114.4437, 455.874],
    "OmniGlue_time": [511.8633, 950.0787, 1778.5049, 3000, np.nan],
    "SemaGlue_memory": [267.354624, 268.93568, 437.084672, 1282.266496, 4583.25248],
    "SGMNet_memory": [275.805952, 290.657792, 339.215872, 703.990784, 2348.616704],
    "SuperGlue_memory": [193.622528, 196.321792, 287.425, 920.797696, 3395.502592],
    "LightGlue_memory": [81.117696, 136.18432, 349.078016, 1177.518592, 4445.01248],
    "OmniGlue_memory": [374.538752, 464.739304, 745.25696, 1590.441984, 4891.424768]
}

df = pd.DataFrame(data)

# 创建绘图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# 定义等距的 x 坐标
x_equal_spacing = np.arange(len(df["Keypoints"]))  # [0, 1, 2, 3, 4]
x_labels = df["Keypoints"]

# 绘制时间图
ax1.plot(x_equal_spacing, df["SemaGlue_time"], 'o-', label="SemaGlue(Ours)", color='orange')
ax1.plot(x_equal_spacing, df["SGMNet_time"], '^-', label="SGMNet", color='purple')
ax1.plot(x_equal_spacing, df["SuperGlue_time"], 'D-', label="SuperGlue", color='green')
ax1.plot(x_equal_spacing, df["LightGlue_time"], 'x-', label="LightGlue", color='cyan')
ax1.plot(x_equal_spacing, df["OmniGlue_time"], 'p-', label="OmniGlue", color='blue')

ax1.set_ylim(0, 2000)
# 设置时间图的标签和标题字体大小
ax1.set_xlabel('Keypoints', fontsize=30)
ax1.set_ylabel('Time (ms)', fontsize=30)
ax1.set_title('Time vs Keypoints', fontsize=30)

ax1.set_xticks(x_equal_spacing)
ax1.set_xticklabels(x_labels)  # 使用原始的刻度标签
# ax1.legend()
ax1.legend(loc='upper right')  # 将图例移到左上角
ax1.grid(True)

# 绘制内存图
ax2.plot(x_equal_spacing, df["SemaGlue_memory"], 'o-', label="SemaGlue(Ours)", color='orange')
ax2.plot(x_equal_spacing, df["SGMNet_memory"], '^-', label="SGMNet", color='purple')
ax2.plot(x_equal_spacing, df["SuperGlue_memory"], 'D-', label="SuperGlue", color='green')
ax2.plot(x_equal_spacing, df["LightGlue_memory"], 'x-', label="LightGlue", color='cyan')
ax2.plot(x_equal_spacing, df["OmniGlue_memory"], 'p-', label="OmniGlue", color='blue')

ax2.set_ylim(0, 5000)
# 设置内存图的标签和标题字体大小
ax2.set_xlabel('Keypoints', fontsize=30)
ax2.set_ylabel('Memory (MB)', fontsize=30)
ax2.set_title('Memory vs Keypoints', fontsize=30)

ax2.set_xticks(x_equal_spacing)
ax2.set_xticklabels(x_labels)  # 使用原始的刻度标签
ax2.legend()
ax2.grid(True)

# plt.tight_layout()
# plt.savefig('output_equal_spacing.png')
# plt.show()
plt.tight_layout()
plt.savefig('computational_usage1.pdf', format='pdf')
plt.show()

