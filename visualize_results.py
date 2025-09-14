import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager
import skimage.io as io
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 加载图像
original = io.imread('data/membrane/train/image/0.png')
ground_truth = io.imread('data/membrane/train/label/0.png') 
prediction = io.imread('data/membrane/test/0_predict.png')

# 创建对比图
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(original, cmap='gray')
axes[0].set_title('原始图像')  # 现在这里应该能正常显示中文了
axes[0].axis('off')

axes[1].imshow(ground_truth, cmap='gray')
axes[1].set_title('真实标注')
axes[1].axis('off')

axes[2].imshow(prediction, cmap='gray')
axes[2].set_title('模型预测')
axes[2].axis('off')

plt.savefig('results_comparison.png', dpi=300, bbox_inches='tight')
plt.show()