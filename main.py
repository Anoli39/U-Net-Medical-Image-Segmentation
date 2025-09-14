# 导入模型定义（从model.py中导入unet函数）
from model import *
# 导入数据处理工具（从data.py中导入数据生成器等函数）
from data import *

# 设置使用的GPU设备（0表示第一块GPU，注释掉则默认使用所有可用GPU/CPU）
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


#1. 数据增强参数配置
# 定义图像增强的参数字典，用于增加训练数据多样性，减轻过拟合
data_gen_args = dict(
    rotation_range=0.2,         # 随机旋转角度范围（0.2度）
    width_shift_range=0.05,     # 水平平移范围（占图像宽度的5%）
    height_shift_range=0.05,    # 垂直平移范围（占图像高度的5%）
    shear_range=0.05,           # 随机剪切变换范围（0.05弧度）
    zoom_range=0.05,            # 随机缩放范围（±5%）
    horizontal_flip=True,       # 随机水平翻转（左右翻转）
    fill_mode='nearest'         # 图像变换后空白区域的填充方式（最近邻填充）
)

# 创建训练数据生成器
# 参数说明：
# 2：批量大小（每次迭代输入2张图像）
# 'data/membrane/train'：训练数据根目录
# 'image'：训练图像所在的子文件夹名
# 'label'：训练标签（掩码）所在的子文件夹名
# data_gen_args：上面定义的数据增强参数
# save_to_dir = None：不保存增强后的图像（若需查看增强效果，可设置为具体路径如"./augmented"）
myGene = trainGenerator(2, 'data/membrane/train', 'image', 'label', data_gen_args, save_to_dir=None)


# 2. 模型构建与训练 
# 构建U-Net模型（默认输入尺寸为(256,256,1)，即256x256的灰度图）
model = unet()

# 定义模型保存回调函数：监控损失值，保存训练过程中损失最小的模型
# 参数说明：
# 'unet_membrane.hdf5'：保存的模型文件名
# monitor='loss'：监控训练损失
# verbose=1：保存模型时打印信息
# save_best_only=True：只保存损失最小的模型（而非每个epoch都保存）
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True)

# 训练模型
# 参数说明：
# myGene：训练数据生成器（源源不断提供增强后的训练数据）
# steps_per_epoch=300：每个epoch包含300个训练步骤（300*batch_size=600张图像/epoch）
# epochs=1：训练总轮数（实际应用中需设置更大值如100）
# callbacks=[model_checkpoint]：训练过程中调用的回调函数（此处用于保存最佳模型）
model.fit_generator(myGene, steps_per_epoch=300, epochs=1, callbacks=[model_checkpoint])


# 3. 模型预测与结果保存 
# 创建测试数据生成器：加载测试集图像并预处理（标准化、尺寸调整等）
# 参数"data/membrane/test"：测试图像所在目录
testGene = testGenerator("data/membrane/test")

# 使用训练好的模型对测试集进行预测
# 参数说明：
# testGene：测试数据生成器
# 30：测试图像总数（需与实际测试集数量一致）
# verbose=1：显示预测进度信息
results = model.predict_generator(testGene, 30, verbose=1)

# 将预测结果保存为图像
# 参数说明：
# "data/membrane/test"：结果保存目录
# results：模型预测输出的结果数组
saveResult("data/membrane/test", results)