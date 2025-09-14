import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *  # 导入Keras模型相关工具（如Model, Input等）
from keras.layers import *  # 导入Keras层相关工具（如卷积层、池化层等）
from keras.optimizers import *  # 导入优化器（如Adam, SGD等）
from keras.callbacks import ModelCheckpoint, LearningRateScheduler  # 导入回调函数（如模型保存、学习率调度）
from keras import backend as keras  # 导入Keras后端（用于底层操作）


def unet(pretrained_weights=None, input_size=(256, 256, 1)):
    """
    构建经典的U-Net图像分割模型
    
    参数:
        pretrained_weights: 预训练权重文件路径（若有则加载）
        input_size: 输入图像尺寸 (高, 宽, 通道数)，默认(256,256,1)为灰度图
    
    返回:
        构建好的U-Net模型
    """
    # 输入层：定义输入图像的尺寸
    inputs = Input(input_size)
    
    # 编码器部分（下采样路径）：通过卷积和池化提取图像特征，缩小空间尺寸
    
    # 第一层卷积：64个3x3卷积核，ReLU激活，same padding（保持尺寸），He正态分布初始化权重
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    # 再次卷积：加深特征提取（U-Net每个阶段包含2次卷积）
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    # 最大池化：2x2窗口，步长2，空间尺寸减半（下采样）
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # 第二层卷积：128个3x3卷积核（特征通道数翻倍）
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # 第三层卷积：256个3x3卷积核
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # 第四层卷积：512个3x3卷积核
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    # Dropout层：随机丢弃50%的神经元，防止过拟合
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    # 第五层（最底层）卷积：1024个3x3卷积核（特征提取能力最强）
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    # 解码器部分（上采样路径）：通过上采样和跳跃连接恢复空间尺寸，融合高低层特征
    
    # 第一层上采样：将特征图尺寸翻倍（2x2上采样），并用512个2x2卷积核调整通道数
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5)  # 上采样操作
    )
    # 跳跃连接：融合编码器对应层（conv4）的高分辨率特征与当前解码器特征
    merge6 = concatenate([drop4, up6], axis=3)  # 沿通道维度拼接（axis=3对应通道维度）
    # 卷积细化：融合后用512个3x3卷积核提取特征
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    
    # 第二层上采样：256个卷积核
    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6)
    )
    merge7 = concatenate([conv3, up7], axis=3)  # 与编码器conv3层融合
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    
    # 第三层上采样：128个卷积核
    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7)
    )
    merge8 = concatenate([conv2, up8], axis=3)  # 与编码器conv2层融合
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    
    # 第四层上采样：64个卷积核
    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8)
    )
    merge9 = concatenate([conv1, up9], axis=3)  # 与编码器conv1层融合
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    
    # 输出层前的卷积：2个卷积核（为二分类做准备）
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    
    # 最终输出层：1个卷积核，sigmoid激活（输出[0,1]范围的概率图，用于二分类分割）
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    
    # 构建模型：定义输入和输出
    model = Model(inputs=inputs, outputs=conv10)
    
    # 模型编译：配置优化器、损失函数和评估指标
    model.compile(
        optimizer=Adam(lr=1e-4),  # Adam优化器，初始学习率1e-4
        loss='binary_crossentropy',  # 二值交叉熵损失（适用于二分类分割）
        metrics=['accuracy']  # 评估指标：准确率
    )
    
    # 打印模型结构（可选，取消注释可查看）
    # model.summary()
    
    # 加载预训练权重（若有）
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    
    return model

