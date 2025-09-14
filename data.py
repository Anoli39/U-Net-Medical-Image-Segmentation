from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans

Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])


def adjustData(img, mask, flag_multi_class, num_class):
    # 1. 多分类分割场景（如12类场景分割）
    if(flag_multi_class):
        # 图像标准化：像素值从[0,255]缩放到[0,1]（神经网络输入需归一化）
        img = img / 255
        
        # 处理掩码：若掩码是4维（batch, h, w, 1），取最后一维（去掉通道维度）
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        
        # 掩码转为“独热编码（One-Hot）”：将单通道类别标签（如像素值=1代表建筑）转为多通道
        # 例如：原mask.shape=(batch, 256, 256) → 新mask.shape=(batch, 256, 256, num_class)
        new_mask = np.zeros(mask.shape + (num_class,))  # 初始化多通道掩码
        
        # 遍历每个类别，将原掩码中“等于该类别”的像素位置，在新掩码对应通道设为1
        for i in range(num_class):
            new_mask[mask == i, i] = 1  # 例如：mask中=0的像素（天空），在new_mask第0通道设为1
        
        # 调整维度：若多分类，将掩码展平为（batch, h*w, num_class）（适配部分分割损失函数输入格式）
        new_mask = np.reshape(new_mask,(new_mask.shape[0], new_mask.shape[1]*new_mask.shape[2], new_mask.shape[3])) 
        mask = new_mask  # 更新掩码为独热编码格式
    
    # 2. 二分类分割场景（如“目标/背景”分割）
    elif(np.max(img) > 1):
        # 图像标准化：[0,255]→[0,1]
        img = img / 255
        # 掩码标准化+二值化：先缩放到[0,1]，再用0.5阈值区分“目标（1）”和“背景（0）”
        mask = mask / 255
        mask[mask > 0.5] = 1  # 超过0.5的像素视为目标
        mask[mask <= 0.5] = 0 # 小于等于0.5的像素视为背景
    
    # 返回处理后的图像和掩码
    return (img, mask)



def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, 
                    image_color_mode="grayscale", mask_color_mode="grayscale",
                    image_save_prefix="image", mask_save_prefix="mask",
                    flag_multi_class=False, num_class=2, save_to_dir=None, 
                    target_size=(256,256), seed=1):
    '''
    参数说明：
    - batch_size：每次生成的样本数量（如32、64，根据显存调整）
    - train_path：训练集根目录（如"./data/train"）
    - image_folder：训练图像子文件夹名（如"images"，路径为train_path/images）
    - mask_folder：训练掩码子文件夹名（如"masks"，路径为train_path/masks）
    - aug_dict：图像增强参数字典（如旋转、翻转、缩放等）
    - image_color_mode/mask_color_mode：图像/掩码颜色模式（"grayscale"灰度，"rgb"彩色）
    - image_save_prefix/mask_save_prefix：增强后图像/掩码的保存前缀（如"image_aug"）
    - save_to_dir：增强后的图像/掩码保存路径（None则不保存，仅用于可视化调试）
    - target_size：图像/掩码缩放后的尺寸（如(256,256)，统一输入大小）
    - seed：随机种子（确保图像和掩码的增强变换完全一致）
    '''
    
    # 1. 初始化图像增强生成器（用aug_dict定义的规则增强）
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)  # 掩码增强器与图像完全一致
    
    # 2. 加载训练图像（仅加载图像，不分类，故class_mode=None）
    image_generator = image_datagen.flow_from_directory(
        train_path,               # 根目录
        classes=[image_folder],   # 仅加载image_folder子文件夹的图像
        class_mode=None,          # 无类别标签（分割任务中图像的“标签”是掩码）
        color_mode=image_color_mode,  # 颜色模式
        target_size=target_size,  # 缩放至目标尺寸
        batch_size=batch_size,    # 批量大小
        save_to_dir=save_to_dir,  # 增强后图像保存路径（None则不保存）
        save_prefix=image_save_prefix,  # 保存文件名前缀
        seed=seed                 # 随机种子（与掩码生成器一致）
    )
    
    # 3. 加载训练掩码（逻辑与图像加载完全一致，确保同步增强）
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed  # 关键：与image_generator同种子，保证变换同步
    )
    
    # 4. 合并图像和掩码生成器（每次返回1个batch的(img, mask)）
    train_generator = zip(image_generator, mask_generator)
    
    # 5. 对每个batch的图像和掩码调用adjustData预处理，然后返回
    for (img, mask) in train_generator:
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        yield (img, mask)  # 生成器：每次迭代返回预处理后的batch数据

def testGenerator(test_path, num_image=30, target_size=(256,256), 
                  flag_multi_class=False, as_gray=True):
    '''
    参数说明：
    - test_path：测试集图像目录（如"./data/test"）
    - num_image：测试图像总数（需与实际测试集数量一致）
    - as_gray：是否按灰度图读取
    '''
    # 遍历每个测试图像（假设测试图像命名为0.png、1.png、...、num_image-1.png）
    for i in range(num_image):
        # 1. 读取测试图像（按灰度/彩色模式）
        img = io.imread(os.path.join(test_path, "%d.png"%i), as_gray=as_gray)
        
        # 2. 标准化：像素值[0,255]→[0,1]
        img = img / 255
        
        # 3. 缩放图像至目标尺寸（与训练输入一致）
        img = trans.resize(img, target_size)
        
        # 4. 调整通道维度：若为二分类/灰度图，添加通道维度（如(256,256)→(256,256,1)）
        if (not flag_multi_class):
            img = np.reshape(img, img.shape + (1,))
        
        # 5. 调整batch维度：模型输入需为(batch, h, w, c)，故添加batch维度（如(256,256,1)→(1,256,256,1)）
        img = np.reshape(img, (1,) + img.shape)
        
        # 生成处理后的单张测试图像（每次返回1张，适配模型预测）
        yield img


def geneTrainNpy(image_path, mask_path, flag_multi_class=False, num_class=2,
                 image_prefix="image", mask_prefix="mask", image_as_gray=True, mask_as_gray=True):
    '''
    参数说明：
    - image_path：训练图像目录（如"./data/train/images"）
    - mask_path：训练掩码目录（如"./data/train/masks"）
    - image_prefix：图像文件名前缀（如"image_0.png"的前缀是"image_"）
    - mask_prefix：掩码文件名前缀（需与图像对应，如"mask_0.png"）
    '''
    # 1. 获取所有图像文件路径（按前缀匹配，如"image*.png"）
    image_name_arr = glob.glob(os.path.join(image_path, "%s*.png"%image_prefix))
    
    # 2. 初始化数组，存储所有图像和掩码
    image_arr = []
    mask_arr = []
    
    # 3. 遍历每个图像，读取并预处理
    for index, item in enumerate(image_name_arr):
        # 读取图像（灰度/彩色），并添加通道维度（如(256,256)→(256,256,1)）
        img = io.imread(item, as_gray=image_as_gray)
        if image_as_gray:
            img = np.reshape(img, img.shape + (1,))
        
        # 读取对应掩码（替换路径和前缀，确保图像与掩码一一对应）
        # 例：item是"./images/image_0.png" → 掩码路径是"./masks/mask_0.png"
        mask = io.imread(item.replace(image_path, mask_path).replace(image_prefix, mask_prefix), 
                        as_gray=mask_as_gray)
        if mask_as_gray:
            mask = np.reshape(mask, mask.shape + (1,))
        
        # 调用adjustData预处理图像和掩码
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        
        # 添加到数组列表
        image_arr.append(img)
        mask_arr.append(mask)
    
    # 4. 转换为numpy数组（便于模型直接读取）
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    
    # 返回图像和掩码数组（shape分别为(batch, h, w, c)和对应掩码格式）
    return image_arr, mask_arr


def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255



def saveResult(save_path, npyfile):
    for i, item in enumerate(npyfile):
        img = item[:, :, 0]
        
        # 添加的代码开始
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255).astype(np.uint8)
        # 添加的代码结束
        
        io.imsave(os.path.join(save_path, "%d_predict.png" % i), img)