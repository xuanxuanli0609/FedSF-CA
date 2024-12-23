import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
# 设置随机数种子
np.random.seed(42)
#E:\Federated_Learning_for_Segmentation\2D\dataset\CHAOS_AMOS22\train\real_label\0
mask =  cv2.imread('E:/Federated_Learning_for_Segmentation/2D/dataset/CHAOS_AMOS22/train/real_label/0/31_image_T2SPIR_21.png', cv2.IMREAD_GRAYSCALE)
# mask =  cv2.imread('E:/Federated_Learning_for_Segmentation/2D/dataset/Inconsistent_Labels_CHAOS_AMOS22/test/real_label/0/31_image_T2SPIR_21.png', cv2.IMREAD_GRAYSCALE)
print(mask.shape)
mask[mask==2]=0
# plt.imshow(mask)
# plt.show()
# 将mask数组转换为图像
mask_image = Image.fromarray(mask.astype(np.uint8))
# 获取类别数量
num_classes = np.max(mask)+1
print(num_classes)
# 创建一个颜色映射表，每个类别映射到一个唯一的颜色
# 这里需要确保颜色映射表中的颜色数量至少与类别数相同
# 例如，这里我们创建了一个简单的颜色映射表，每个类别映射到一个不同的颜色
color_map = {i: (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for i in range(num_classes)}

# 创建一个新的彩色图像
colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

# 为每个类别分配颜色
for class_value in range(num_classes):
    if class_value==0:
        continue
    colored_mask[mask == class_value] = color_map[class_value]

# 显示彩色化的图像
cv2.imshow('Colored Mask', colored_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存彩色化的图像
cv2.imwrite('full_31_image_T2SPIR_21.png', colored_mask)