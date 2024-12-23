import os
import cv2
import shutil
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import numpy as np

def rgb2gray(mask):
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int8)
    # 标签处理
    for ii, label in enumerate(COLORMAP):
        # np.all很巧妙，可以自己搜索一下
        locations = np.all(mask == label, axis=-1)
        label_mask[locations] = ii
    return label_mask.astype(np.uint8)

def gray2rgb(label_mask):
    mask = np.zeros((label_mask.shape[0], label_mask.shape[1], 3), dtype=np.int8)
    for i in range(6):
        # 这里也困了很久没相通，结果答案很简单，大家可以自行打印出来看一下中间的东西是什么
        mask[label_mask == i] = COLORMAP[i]
    return mask.astype(np.uint8)

# 定义JSON文件的路径
json_file_path = './EndoVis2018_labels.json'

# 打开文件
with open(json_file_path, 'r', encoding='utf-8') as file:
    # 读取JSON内容并将其转换为Python字典
    labels_map = json.load(file)

# 打印读取的数据
COLORMAP = []
lab_val = []
for map in labels_map:
    COLORMAP.append(map['color'])
    lab_val.append(map['classid'])
COLORMAP = np.array(COLORMAP)

base = '../../EndoVis2018/train'
out_base = '../../dataset/EndoVis2018/train'
client_list = os.listdir(base)
n = 0
copy_im = False
get_lab = False
get_test = False
for c in client_list:
    im_dir_list = os.listdir(os.path.join(base, c))
    #解决图像复制，重命名问题
    if copy_im:
        for im_dir in tqdm(im_dir_list):
            head = im_dir
            im_dir = os.path.join(base,c,im_dir,'left_frames')
            if not os.path.isdir(im_dir):
                continue
            for im_path in os.listdir(im_dir):
                name = im_path
                # 定义源目录和目标目录
                source_dir = os.path.join(im_dir,im_path)
                if not os.path.exists(os.path.join(out_base,'real_image/'+str(n))):
                    os.mkdir(os.path.join(out_base,'real_image/'+str(n)))
                destination_dir = os.path.join(out_base,'real_image/'+str(n),head+'_'+name)
                # 复制整个目录
                shutil.copy(source_dir, destination_dir)
    #标注问题
    if get_lab:
        for im_dir in tqdm(im_dir_list):
            head = im_dir
            im_dir = os.path.join(base, c, im_dir, 'labels')
            if not os.path.isdir(im_dir):
                continue
            # print(im_dir)
            for im_path in os.listdir(im_dir):
                name = im_path
                # 定义源目录和目标目录
                source_dir = os.path.join(im_dir, im_path)
                lab = cv2.imread(source_dir, cv2.IMREAD_COLOR)
                lab = lab[:,:,::-1]
                # 这里先后实现了彩色标签转成pytorch需要的单通道标签
                label_img_gray = rgb2gray(lab)
                # plt.imshow(label_img_gray)
                # plt.show()
                # print(lab.shape)
                if not os.path.exists(os.path.join(out_base, 'real_label/' + str(n))):
                    os.mkdir(os.path.join(out_base, 'real_label/' + str(n)))
                destination_dir = os.path.join(out_base, 'real_label/' + str(n), head + '_' + name)
                cv2.imwrite(destination_dir,label_img_gray)
    n+=1

#设置测试数据
r = 0.2
test_path = out_base.replace('train','test')
im_path = os.path.join(out_base,'real_image')
lab_path = os.path.join(out_base,'real_label')
n = 0
if get_test:
    for c in os.listdir(im_path):
        im_dir = os.path.join(im_path,c)
        im_list = os.listdir(im_dir)
        random.shuffle(im_list)
        test_im_list = im_list[:int(len(im_list)*r)]
        if not os.path.exists(os.path.join(test_path, 'real_image/' + str(n))):
            os.mkdir(os.path.join(test_path, 'real_image/' + str(n)))
        if not os.path.exists(os.path.join(test_path, 'real_label/' + str(n))):
            os.mkdir(os.path.join(test_path, 'real_label/' + str(n)))
        for t in test_im_list:
            source_dir = os.path.join(im_dir, t)
            destination_dir = os.path.join(test_path, 'real_image/' + str(n), t)
            # 复制整个目录
            shutil.copy(source_dir, destination_dir)
            os.remove(source_dir)
            shutil.copy(source_dir.replace('real_image','real_label'), destination_dir.replace('real_image','real_label'))
            os.remove(source_dir.replace('real_image','real_label'))
        n+=1



#设置标签异构。
base = '../../dataset/Inconsistent_Labels_EndoVis2018/train/real_label'
clients = os.listdir(base)
tjs = []
labels_map = {
                '0':[4,6,7],
                '1':[2,3],
                '2':[2,5],
                '3':[1,10], #8，9，11由于类别太少决定删除这三个类别
              }
for c in clients:
    map  = labels_map[c]
    lab_unique = set()
    labs = os.listdir(os.path.join(base,c))
    tj = np.zeros([12])
    for lab in labs:
        # print(os.path.join(base,c,lab))
        lab_arr = cv2.imread(os.path.join(base,c,lab),cv2.IMREAD_GRAYSCALE)
        unique = np.unique(lab_arr)
        for un in unique:
            if un not in map:
                lab_arr[lab_arr==un]=0

        cv2.imwrite(os.path.join(base,c,lab),lab_arr)
        # plt.imshow(lab_arr)
        # plt.show()
        for i in unique:
            tj[i] += 1
        lab_unique = lab_unique | set(np.unique(lab_arr))
        # print(lab_arr.shape)
        # plt.imshow(lab_arr)
        # plt.show()
        # exit()
    tjs.append(tj)

base = '../../dataset/Inconsistent_Labels_EndoVis2018/test/real_label'
clients = os.listdir(base)
tjs = []
labels_map = {
                '0':[1,2,3,4,5,6,7,10],
                '1':[1,2,3,4,5,6,7,10],
                '2':[1,2,3,4,5,6,7,10],
                '3':[1,2,3,4,5,6,7,10], #8，9，11由于类别太少决定删除这三个类别
              }
for c in clients:
    map  = labels_map[c]
    lab_unique = set()
    labs = os.listdir(os.path.join(base,c))
    tj = np.zeros([12])
    for lab in labs:
        # print(os.path.join(base,c,lab))
        lab_arr = cv2.imread(os.path.join(base,c,lab),cv2.IMREAD_GRAYSCALE)
        unique = np.unique(lab_arr)
        for un in unique:
            if un not in map:
                lab_arr[lab_arr==un]=0

        cv2.imwrite(os.path.join(base,c,lab),lab_arr)
        # plt.imshow(lab_arr)
        # plt.show()
        for i in unique:
            tj[i] += 1
        lab_unique = lab_unique | set(np.unique(lab_arr))
        # print(lab_arr.shape)
        # plt.imshow(lab_arr)
        # plt.show()
        # exit()
    tjs.append(tj)


print('      1,      2,      3,      4,      5,      6,      7,      8,      9,     10,     11')
for tj in tjs:
    tj = 100*tj/tj[0]
    for n in tj[1:]:
        print("%5.2f" % n+'%',end=' ,')
    print()