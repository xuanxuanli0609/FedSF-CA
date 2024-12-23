import numpy as np
import pandas as pd
import os
import SimpleITK as sitk
from matplotlib import pyplot as plt
from tqdm import tqdm
import random
import cv2

min_spacing = [1.0,1.0,1.5]

#统计均值和方差，查看分布情况
def read_image(fname):
    reader = sitk.ImageFileReader()
    reader.SetFileName(fname)
    image = reader.Execute()
    #image.SetDirection((1,0,0,0,1,0,0,0,1))
    image = sitk.DICOMOrient(image, "LPS")
    return image

def resample_image(image, spacing_rs, linear):
    size = np.array(image.GetSize())
    origin = np.array(image.GetOrigin())
    spacing = np.array(image.GetSpacing())
    spacing_rs = np.array(spacing_rs)
    size_rs = (size * spacing / spacing_rs).astype(dtype=np.int32)
    origin_rs = origin + 0.5 * size * spacing - 0.5 * size_rs * spacing_rs
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize((int(size_rs[0]), int(size_rs[1]), int(size_rs[2])))
    resampler.SetOutputSpacing((float(spacing_rs[0]), float(spacing_rs[1]), float(spacing_rs[2])))
    resampler.SetOutputOrigin((float(origin_rs[0]), float(origin_rs[1]), float(origin_rs[2])))
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    if linear:
        resampler.SetInterpolator(sitk.sitkLinear)
    else:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    rs_image = resampler.Execute(image)

    return rs_image

def find_files_with_keyword(directory, keyword):
    matching_files = []
    # os.walk遍历directory及其所有子文件夹
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 检查文件名是否包含keyword
            if keyword in file:
                matching_files.append(os.path.join(root, file))  # 将完整路径添加到列表中
    return matching_files


if not os.path.exists('./2D_MM_WHS/train/real_image/1/'):
    os.makedirs('./2D_MM_WHS/train/real_image/1/')
if not os.path.exists('./2D_MM_WHS/train/real_label/1/'):
    os.makedirs('./2D_MM_WHS/train/real_label/1/')
if not os.path.exists('./2D_MM_WHS/train/real_image/0/'):
    os.makedirs('./2D_MM_WHS/train/real_image/0/')
if not os.path.exists('./2D_MM_WHS/train/real_label/0/'):
    os.makedirs('./2D_MM_WHS/train/real_label/0/')
if not os.path.exists('./2D_MM_WHS/test/real_image/1/'):
    os.makedirs('./2D_MM_WHS/test/real_image/1/')
if not os.path.exists('./2D_MM_WHS/test/real_label/1/'):
    os.makedirs('./2D_MM_WHS/test/real_label/1/')
if not os.path.exists('./2D_MM_WHS/test/real_image/0/'):
    os.makedirs('./2D_MM_WHS/test/real_image/0/')
if not os.path.exists('./2D_MM_WHS/test/real_label/0/'):
    os.makedirs('./2D_MM_WHS/test/real_label/0/')

seed = 0
np.random.seed(seed)
random.seed(seed)
# 指定要搜索的文件夹路径
directory_path = 'E:/Federated_Learning_for_Segmentation/heart_class/4_MM-WHS 2017 Dataset/MM-WHS 2017 Dataset/'
# 指定要搜索的关键词
keyword = 'image'
# 获取所有包含关键词的文件的完整路径
image_files = find_files_with_keyword(directory_path, keyword)
# 打印结果
# for image_file in image_files:
#     print(image_file)

#[  0 205 420 500 550 600 820 850]  [  0  38  52  82  88 164 205 244]
lab_map = {
    'ct':{0:0,38:38,52:52,82:82,88:0,164:0,205:0,244:0},
    'mr':{0:0,38:0,52:0,82:0,88:88,164:164,205:205,244:0},
    'label_view':{0:0,38:38,52:52,82:82,88:88,164:164,205:205,244:0},
    'label':{0:0,38:1,52:2,82:3,88:4,164:5,205:6,244:0}
}

dir_list = image_files
random.shuffle(dir_list)
im_mes_vals = []


#20%的概率生成测试图片
get_data = True
get_data_view = False
for d in tqdm(dir_list, desc='Processing', unit='step'):
    if 'test' in d:
        continue
    try:
        print(d)
        im_path = d#os.path.join(base_dir,d,'ct.nii.gz')
        lab_path = d.replace('image','label')
        im = read_image(im_path)
        la = read_image(lab_path)
    except:
        print('erray!')
        continue
    im_arr = sitk.GetArrayFromImage(im) #[x,y,z]
    la_arr = sitk.GetArrayFromImage(la).astype(np.uint8) #[x,y,z]

    la_arr[la_arr==165] = 164


    im_mes_vals.append([np.mean(im_arr), np.var(im_arr)])
    # print(im_arr.shape)
    # print(la_arr.shape)
    # print(np.unique(la_arr))
    # print(im.GetSpacing())
    # print(la.GetSpacing())
    im_shape = im_arr.shape
    if 'ct' in d:
        im_arr[im_arr<-200] = -200
        im_arr[im_arr> 700] = 700
        #查看图像的分布是否为[x,y,z]
        if im_shape[0] == im_shape[1]:#[x,y,z]正常
            pass
        if im_shape[0] == im_shape[2]:#[x,z,y]不正常
            # 重新排序轴，将x轴移到最前面
            im_arr = im_arr.transpose((0, 2, 1))
            la_arr = la_arr.transpose((0, 2, 1))
        if im_shape[1] == im_shape[2]:#[z,x,y]不正常
            # 重新排序轴，将x轴移到最前面
            im_arr = im_arr.transpose((1, 2, 0))
            la_arr = la_arr.transpose((1, 2, 0))
        map = lab_map['ct']
    if 'mr' in d:
        im_arr[im_arr<-800] = -800
        im_arr[im_arr> 1200] = 1200
        im_arr = im_arr.transpose((1,2,0))
        la_arr = la_arr.transpose((1,2,0))
        map = lab_map['mr']
    # 找到最小值和最大值
    # min_val = im_arr.min()
    # max_val = im_arr.max()
    h1 = d.split('\\')[1].split('.')[0]

    for num in range(im_arr.shape[2]):
        lab = la_arr[:,:,num]
        # if 82 in lab:
        #     print(np.unique(lab))
        #     lab[lab!=82] = 0
        #     plt.imshow(lab)
        #     plt.show()
        #     continue
        # else:
        #     continue

        if len(np.unique(lab)) < 7:
            continue
        img = im_arr[:,:,num].astype(np.float32)  # 确保是浮点数
        min_val = img.min()
        max_val = img.max()
        img = (img - min_val) / (max_val - min_val) * 255
        # 转换为uint8类型
        im_arr_channel_scaled_uint8 = img.astype(np.uint8)
        # 将单通道图像转换为三通道彩色图像
        im_color = np.dstack((im_arr_channel_scaled_uint8, im_arr_channel_scaled_uint8, im_arr_channel_scaled_uint8))
        print(np.unique(lab))
        #Label 205	Label 420	Label 500	Label 550	Label 600	Label 820

        random_number = random.random()
        if random_number<=0.2:#0.2的数据为测试数据
            if get_data:
                map_label = lab_map['label']
                for k, v in map_label.items():
                    lab[lab == k] = v
            else:
                map_label = lab_map['label_view']
                for k, v in map_label.items():
                    lab[lab == k] = v
            # print(np.unique(lab))
            # 保存三通道彩色图像
            im_color = cv2.resize(im_color, (512, 512), interpolation=cv2.INTER_AREA)
            lab = cv2.resize(lab, (512, 512), interpolation=cv2.INTER_AREA)
            if 'mr' in d:
                cv2.imwrite('./2D_MM_WHS/test/real_image/1/' + h1 + f'_{num}.png', im_color)
                cv2.imwrite('./2D_MM_WHS/test/real_label/1/' + h1 + f'_{num}.png', lab)
            if 'ct' in d:
                cv2.imwrite('./2D_MM_WHS/test/real_image/0/' + h1 + f'_{num}.png', im_color)
                cv2.imwrite('./2D_MM_WHS/test/real_label/0/' + h1 + f'_{num}.png', lab)
        else:
            for k, v in map.items():
                lab[lab == k] = v
            if get_data==True:
                # print('nnnnnnnnnnnnnnnnnnnn')
                map_label = lab_map['label']
                for k, v in map_label.items():
                    # print(k, v)
                    lab[lab == k] = v
            print(np.unique(lab))
            # 保存三通道彩色图像
            im_color = cv2.resize(im_color, (512, 512), interpolation=cv2.INTER_AREA)
            lab = cv2.resize(lab, (512, 512), interpolation=cv2.INTER_AREA)
            if 'mr' in d:
                cv2.imwrite('./2D_MM_WHS/train/real_image/1/'+h1+f'_{num}.png', im_color)
                cv2.imwrite('./2D_MM_WHS/train/real_label/1/'+h1+f'_{num}.png', lab)
            if 'ct' in d:
                cv2.imwrite('./2D_MM_WHS/train/real_image/0/'+h1+f'_{num}.png', im_color)
                cv2.imwrite('./2D_MM_WHS/train/real_label/0/' + h1 + f'_{num}.png', lab)

    # exit()


    # if len(im_mes_vals) >= 100:
    #     break

im_mes_vals = np.array(im_mes_vals)
print(im_mes_vals.shape)
# 创建一个新的图和坐标轴
fig, ax = plt.subplots(figsize=(10, 20))
# 遍历每个数组并绘制点
ax.scatter(im_mes_vals[:, 0], im_mes_vals[:, 1], c='y', label=f'Set {1}', s=2)
ax.legend()
# 设置坐标轴的标题
ax.set_title('Scatter Plot of Points from Multiple Arrays')
# 设置坐标轴的标签
ax.set_xlabel('X coordinate')
ax.set_ylabel('Y coordinate')
# 显示图像
plt.show()


# import os
# import random
# import shutil
# im_base = 'E:/Federated_Learning_for_Segmentation/2D/FedST_dataset/MM_WHS/train/real_image/1/'
# lab_base = 'E:/Federated_Learning_for_Segmentation/2D/FedST_dataset/MM_WHS/train/real_label/1/'
#
# im_out_base = 'E:/Federated_Learning_for_Segmentation/2D/FedST_dataset/MM_WHS/test/real_image/1/'
# lab_out_base = 'E:/Federated_Learning_for_Segmentation/2D/FedST_dataset/MM_WHS/test/real_label/1/'
# if not os.path.exists(im_out_base):
#     os.makedirs(im_out_base)
#
# if not os.path.exists(lab_out_base):
#     os.makedirs(lab_out_base)
#
# im_list = os.listdir(im_base)
# random.shuffle(im_list)
#
# for i,im_p in enumerate(im_list[:127]):
#     # 源文件路径
#     src_im_path = im_base+im_p
#     src_lab_path = lab_base+im_p
#
#     # 目标文件夹路径
#     dest_im_path = im_out_base+im_p
#     dest_lab_path = lab_out_base+im_p
#
#     # 移动文件
#     try:
#         shutil.move(src_im_path, dest_im_path)
#         shutil.move(src_lab_path, dest_lab_path)
#         print("文件移动成功。")
#     except Exception as e:
#         print(f"文件移动失败：{e}")

#
# label_base = 'E:/Federated_Learning_for_Segmentation/2D/FedST_dataset/2D_MM_WHS/CT_first/real_label_all_label/'
# label_out = 'E:/Federated_Learning_for_Segmentation/2D/FedST_dataset/MM_WHS/train/real_label/0/'
#
# # label_base = 'E:/Federated_Learning_for_Segmentation/2D/FedST_dataset/2D_MM_WHS/MR_first/real_label_all_label/'
# # label_out = 'E:/Federated_Learning_for_Segmentation/2D/FedST_dataset/MM_WHS/train/real_label/1/'
# lab_map = {
#     'ct':{0:0,38:1,52:2,82:3,88:4,164:5,205:6,244:0},
#     'mr':{0:0,38:1,52:2,82:3,88:4,164:5,205:6,244:0}
# }
#

#
# lab_map = lab_map['ct']
# lab_list = os.listdir(label_out)
# for la in lab_list:
#     in_path = os.path.join(label_base,la)
#     out_path = os.path.join(label_out,la)
#     print(in_path)
#     la = cv2.imread(in_path,cv2.IMREAD_GRAYSCALE)
#     plt.imshow(la)
#     plt.show()
#     for k,v in lab_map.items():
#         print(np.unique(la))
#         # if v == 0:
#         #     print(1)
#         #     la[la==k]=0
#         la[la==k] = v
#     plt.imshow(la)
#     plt.show()
#     la = cv2.resize(la, (512, 512), interpolation=cv2.INTER_AREA)
#     cv2.imwrite(out_path, la)


la = cv2.imread(r'E:\Federated_Learning_for_Segmentation\2D\FedST_dataset\MM_WHS\test\real_label\0\ct_train_1001_image_175.png',cv2.IMREAD_GRAYSCALE)
print(np.unique(la))
plt.imshow(la)
plt.show()