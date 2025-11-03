import matplotlib.pyplot as plt
import pandas as pd

# # 创建数据框架
# data = {
#     "Keypoints": [512, 1024, 2048, 4096, 8192],
#     "SGFM_time": [60.6620, 60.0693, 61.0891, 143.3862, 530.6532],
#     "SGFM_7_layers_time": [51.2441, 51.5303, 52.1873, 118.0647, 418.3446],
#     "SGMNet_time": [49.8407, 50.0736, 51.9985, 65.5233, 120.8675],
#     "SuperGlue_time": [27.6999, 28.0172, 43.8130, 153.5748, 677.1354],
#     "LightGlue_noprune_time": [17.2130, 17.3357, 25.1506, 86.5754, 374.1752],
#     "SGFM_memory": [173.576704, 173.978112, 314.110464, 1129.411072, 4371.378688],
#     "SGFM_7_layers_memory": [172.227072, 172.62848, 312.760832, 1128.06144, 4370.029056],
#     "SGMNet_memory": [155.563098, 166.414848, 206.99904, 562.515968, 2181.976064],
#     "SuperGlue_memory": [146.791936, 147.52512, 220.540416, 831.892992, 3262.557696],
#     "LightGlue_noprune_memory": [31.364608, 83.154432, 287.397376, 1098.536448, 4445.01248]
# }

# df = pd.DataFrame(data)

# # 创建绘图
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# # 绘制时间图
# ax1.plot(df["Keypoints"], df["SGFM_time"], 'o-', label="SGFM(Ours)", color='orange')
# ax1.plot(df["Keypoints"], df["SGFM_7_layers_time"], 's-', label="SGFM(Ours) 7 Layers", color='red')
# ax1.plot(df["Keypoints"], df["SGMNet_time"], '^-', label="SGMNet", color='purple')
# ax1.plot(df["Keypoints"], df["SuperGlue_time"], 'D-', label="SuperGlue", color='green')
# ax1.plot(df["Keypoints"], df["LightGlue_noprune_time"], 'x-', label="LightGlue Noprune", color='cyan')
# ax1.set_xlabel('Keypoints')
# ax1.set_ylabel('Time (ms)')
# ax1.set_title('Time vs Keypoints')
# ax1.set_xticks([512, 1024, 2048, 4096, 8192])
# ax1.legend()
# ax1.grid(True)

# # 绘制内存图
# ax2.plot(df["Keypoints"], df["SGFM_memory"], 'o-', label="SGFM(Ours)", color='orange')
# ax2.plot(df["Keypoints"], df["SGFM_7_layers_memory"], 's-', label="SGFM(Ours) 7 Layers", color='red')
# ax2.plot(df["Keypoints"], df["SGMNet_memory"], '^-', label="SGMNet", color='purple')
# ax2.plot(df["Keypoints"], df["SuperGlue_memory"], 'D-', label="SuperGlue", color='green')
# ax2.plot(df["Keypoints"], df["LightGlue_noprune_memory"], 'x-', label="LightGlue Noprune", color='cyan')
# ax2.set_xlabel('Keypoints')
# ax2.set_ylabel('Memory (MB)')
# ax2.set_title('Memory vs Keypoints')
# ax2.set_xticks([512, 1024, 2048, 4096, 8192])
# ax2.legend()
# ax2.grid(True)

# plt.tight_layout()
# plt.savefig('output.png')
# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# # 创建数据框架
# data = {
#     "Keypoints": [512, 1024, 2048, 4096, 8192],
#     "SemaGlue_time": [60.5502, 61.4625, 84.5645, 208.3004, 714.5935],
#     # "SGFM_7_layers_time": [52.3253, 52.0749, 70.1488, 166.6591, 557.838],
#     "SGMNet_time": [50.8007, 50.6555, 51.7695, 66.5256, 127.7146],
#     "SuperGlue_time": [27.4191, 27.6679, 49.002, 169.7118, 742.368],
#     "LightGlue_time": [17.3984, 17.4153, 36.2952, 114.4437, 455.874],
#     "OmniGlue_time": [511.8633, 950.0787, 1778.5049, 3000, np.nan],
#     "SemaGlue_memory": [267.354624, 268.93568, 437.084672, 1282.266496, 4583.25248],
#     # "SGFM_7_layers_memory": [246.275584, 247.85664, 416.005632, 1261.190656, 4562.17344],
#     "SGMNet_memory": [275.805952, 290.657792, 339.215872, 703.990784, 2348.616704],
#     "SuperGlue_memory": [193.622528, 196.321792, 287.425, 920.797696, 3395.502592],
#     "LightGlue_memory": [81.117696, 136.18432, 349.078016, 1177.518592, 4445.01248],
#     "OmniGlue_memory": [374.538752, 464.739304, 745.25696, 1590.441984, 4891.424768]
# }

# df = pd.DataFrame(data)

# # 创建绘图
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# # 绘制时间图
# ax1.plot(df["Keypoints"], df["SemaGlue_time"], 'o-', label="SemaGlue(Ours)", color='orange')
# # ax1.plot(df["Keypoints"], df["SGFM_7_layers_time"], 's-', label="SGFM(Ours) 7 Layers", color='red')
# ax1.plot(df["Keypoints"], df["SGMNet_time"], '^-', label="SGMNet", color='purple')
# ax1.plot(df["Keypoints"], df["SuperGlue_time"], 'D-', label="SuperGlue", color='green')
# ax1.plot(df["Keypoints"], df["LightGlue_time"], 'x-', label="LightGlue Noprune", color='cyan')
# ax1.plot(df["Keypoints"], df["OmniGlue_time"], 'p-', label="OmniGlue", color='blue')

# # 标记超出范围的点
# #ax1.annotate('Out of Range', (4096, 2000), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=12, color='blue')

# ax1.set_ylim(0, 2000)
# ax1.set_xlabel('Keypoints')
# ax1.set_ylabel('Time (ms)')
# ax1.set_title('Time vs Keypoints')
# ax1.set_xticks([512, 1024, 2048, 4096, 8192])
# ax1.legend()
# ax1.grid(True)

# # 绘制内存图
# ax2.plot(df["Keypoints"], df["SemaGlue_memory"], 'o-', label="SemaGlue(Ours)", color='orange')
# # ax2.plot(df["Keypoints"], df["SGFM_7_layers_memory"], 's-', label="SGFM(Ours) 7 Layers", color='red')
# ax2.plot(df["Keypoints"], df["SGMNet_memory"], '^-', label="SGMNet", color='purple')
# ax2.plot(df["Keypoints"], df["SuperGlue_memory"], 'D-', label="SuperGlue", color='green')
# ax2.plot(df["Keypoints"], df["LightGlue_memory"], 'x-', label="LightGlue Noprune", color='cyan')
# ax2.plot(df["Keypoints"], df["OmniGlue_memory"], 'p-', label="OmniGlue", color='blue')

# # 标记超出范围的点
# #ax2.annotate('Out of Range', (4096, 5000), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=12, color='blue')
# #ax2.annotate('Out of Range', (5101, 5000), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=12, color='blue')

# ax2.set_ylim(0, 5000)
# ax2.set_xlabel('Keypoints')
# ax2.set_ylabel('Memory (MB)')
# ax2.set_title('Memory vs Keypoints')
# ax2.set_xticks([512, 1024, 2048, 4096, 8192])
# ax2.legend()
# ax2.grid(True)

# plt.tight_layout()
# plt.savefig('output255.png')
# plt.show()


#--------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置字体大小
plt.rcParams.update({
    'font.size': 14,         # 全局字体大小
    'axes.titlesize': 22,    # 标题字体大小
    'axes.labelsize': 16,    # 轴标签字体大小
    'xtick.labelsize': 14,   # x轴刻度字体大小
    'ytick.labelsize': 14,   # y轴刻度字体大小
    'legend.fontsize': 16    # 图例字体大小
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

# 绘制时间图
ax1.plot(df["Keypoints"], df["SemaGlue_time"], 'o-', label="SemaGlue(Ours)", color='orange')
ax1.plot(df["Keypoints"], df["SGMNet_time"], '^-', label="SGMNet", color='purple')
ax1.plot(df["Keypoints"], df["SuperGlue_time"], 'D-', label="SuperGlue", color='green')
ax1.plot(df["Keypoints"], df["LightGlue_time"], 'x-', label="LightGlue", color='cyan')
ax1.plot(df["Keypoints"], df["OmniGlue_time"], 'p-', label="OmniGlue", color='blue')

ax1.set_ylim(0, 2000)
ax1.set_xlabel('Keypoints')
ax1.set_ylabel('Time (ms)')
ax1.set_title('Time vs Keypoints')
ax1.set_xticks([512, 1024, 2048, 4096, 8192])
ax1.legend()
ax1.grid(True)

# 绘制内存图
ax2.plot(df["Keypoints"], df["SemaGlue_memory"], 'o-', label="SemaGlue(Ours)", color='orange')
ax2.plot(df["Keypoints"], df["SGMNet_memory"], '^-', label="SGMNet", color='purple')
ax2.plot(df["Keypoints"], df["SuperGlue_memory"], 'D-', label="SuperGlue", color='green')
ax2.plot(df["Keypoints"], df["LightGlue_memory"], 'x-', label="LightGlue", color='cyan')
ax2.plot(df["Keypoints"], df["OmniGlue_memory"], 'p-', label="OmniGlue", color='blue')

ax2.set_ylim(0, 5000)
ax2.set_xlabel('Keypoints')
ax2.set_ylabel('Memory (MB)')
ax2.set_title('Memory vs Keypoints')
ax2.set_xticks([512, 1024, 2048, 4096, 8192])
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('output256.png')
plt.show()

