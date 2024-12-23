import argparse
import os

import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torchvision.transforms as tr
from collections import OrderedDict

np.random.seed(42)
# --dataset face --fold 0 --model_path $model_path --algorithm fedst_separate
parser = argparse.ArgumentParser()
# CHAOS_liver   Material   Face_segment
parser.add_argument('--dataset', default='Inconsistent_CT&MR',
                    choices=['material', 'medical', 'medical_material',
                             'face', 'aaf_face', 'Inconsistent_Labels_face',
                             'EndoVis2018', 'Inconsistent_EndoVis2018', 'CT&MR', 'Inconsistent_CT&MR'],
                    help='which dataset to use')
parser.add_argument('--model_path', type=str,
                    # E:\Federated_Learning_for_Segmentation\2D\dataset\CHAOS_AMOS22\model_fedavg_1016_175802,
                    # default='E:/Federated_Learning_for_Segmentation/2D/FedST_dataset/Inconsistent_Labels_face_segment/model_fedddpm_0815_161538/model149_folds',
                    #E:\Federated_Learning_for_Segmentation\2D\dataset\Inconsistent_Labels_CHAOS_AMOS22\model_fedavg_1016_213258
                    #model_fedavg_1016_235047_model99_folds_0_eval.txt
                    #model_feddc_0905_214939 66 199
                    # default='E:/Federated_Learning_for_Segmentation/2D/dataset/face_segment/model_fedddpm_1201_193733/model96_folds',
                    default='E:/Federated_Learning_for_Segmentation/2D/dataset/face_segment/model_fedddpm_1201_193733/model96_folds',
                    help='Choose which model to use,unet...')#E:\Federated_Learning_for_Segmentation\2D\dataset\CT Lung & Heart & Trachea segmentation\model_fedavg_1010_132033
parser.add_argument('--fold', type=str, default='0', help='chooses fold...')
parser.add_argument('--algorithm', type=str, default='fedavg',
                    help='Chooses which federated learning algorithm to use')
parser.add_argument('--client_id', type=int, default=-1,
                    help='Client id for local trainer')
parser.add_argument('--bg_value', type=int, default=0,
                    help='Background pixel value for evaluating test set')
parser.add_argument('--save_image', type=int, default=1,
                    help='Whether to save the predicted mask image')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids:e.g.0 0,1,2 1,2 use -1 for CPU')
parser.add_argument('--use_san_saw', type=bool, default=True, help='is use SAN-SAW model')
op = parser.parse_args()
dataset = op.dataset
if dataset == 'material':
    from opt.material_options import TrainOptions, TestOptions
elif dataset == 'face':
    from opt.face_options import TrainOptions, TestOptions
elif dataset == 'medical':
    from opt.medical_options import TrainOptions, TestOptions
elif op.dataset == 'medical_material':
    from opt.medical_material_options import TrainOptions, TestOptions
elif op.dataset == 'aaf_face':
    from opt.aaf_face_options import TrainOptions, TestOptions
elif op.dataset == 'Inconsistent_Labels_face':
    from opt.Inconsistent_Labels_face_options import TrainOptions, TestOptions
elif op.dataset == 'MM_WHS':
    from opt.MM_WHS import TrainOptions, TestOptions
elif op.dataset == 'EndoVis2018':
    from opt.EndoVis2018 import TrainOptions, TestOptions
elif op.dataset == 'Inconsistent_EndoVis2018':
    from opt.Inconsistent_Labels_EndoVis2018 import TrainOptions, TestOptions
elif op.dataset == 'CT&MR':
    from opt.CT_MR import TrainOptions, TestOptions
elif op.dataset == 'Inconsistent_CT&MR':
    from opt.Inconsistent_Labels_CT_MR import TrainOptions, TestOptions
fold = op.fold
fed_methods = op.algorithm
model_path = op.model_path + fold + '.pkl'
method = dataset + '_' + fed_methods + '_fold_' + fold
client_id = op.client_id
bg_value = op.bg_value
from data.dataloader.data_loader import CreateDataLoader
from models import create_model
import time
from util.util import mkdir
from util.metrics import get_total_evaluation
import torch
import tqdm

is_save_image = True if op.save_image>0 else False

if __name__ == '__main__':
    gpu_ids = op.gpu_ids.split(',')
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in gpu_ids])
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    opt = TestOptions().parse()
    opt_train = TrainOptions().parse()

    # hard-code some parameters for test
    opt.num_threads = 1  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.gpu_ids = [i for i in range(len(gpu_ids))]  # no flip
    opt_train.gpu_ids = [i for i in range(len(gpu_ids))]  # no flip
    opt.client_id = client_id
    opt.use_san_saw = op.use_san_saw
    opt_train.use_san_saw = op.use_san_saw

    if opt.no_normalize:
        transform = tr.ToTensor()
    else:
        transform = tr.Compose([tr.ToTensor(),
                                tr.Normalize(mean=0.5,
                                             std=0.5)
                                ])
    data_loader = CreateDataLoader(opt, dataroot=opt.dataroot, image_dir=opt.test_img_dir, \
                                   label_dir=opt.test_label_dir, record_txt=None, transform=transform, is_aug=False)
    print(opt.gpu_ids)
    print(opt.dataroot)
    print(opt.test_img_dir)
    print(opt.test_label_dir)

    dataset = data_loader.load_data()
    datasize = len(data_loader)
    print('#test images = %d, batchsize = %d' % (datasize, opt.batch_size))
    state_dict = torch.load(model_path, map_location='cuda:0')
    sd = OrderedDict()

    # opt.isTrain = True
    opt_train.model = 'unet'
    model = create_model(opt_train)
    try:
        model.load_state_dict(state_dict)
    except:
        new_state_dict = {}
        for nkey in state_dict.keys():
            new_key = nkey.replace('.module','')
            new_state_dict[new_key] = state_dict[nkey]
        model.load_state_dict(new_state_dict)
    print(f"[Test] Model {model_path} has loaded.")

    # img_dir = os.path.join(opt.results_dir, opt.name, '%s' % method)
    save_dir_name = '_'.join(str(op.model_path).split('/')[-2:]) + '_' + fold
    img_dir = os.path.join(opt.results_dir, opt.name, save_dir_name)
    print('Eval result has saved to %s' % img_dir)
    mkdir(img_dir)

    eval_results = {}
    count = 0
    with open(img_dir + '_eval.txt', 'w') as log:
        now = time.strftime('%c')
        log.write('=============Evaluation (%s)=============\n' % now)

    result = []

    color_map = {i: (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for i in
                 range(opt.output_nc)}
    for i, data in enumerate(tqdm.tqdm(dataset)):
        c_idx = 0

        model.set_input(data)
        with torch.no_grad():
            pred = model()[0]

        count += 1

        pred = pred.max(0)[1].cpu().numpy()
        mask = data['label'].squeeze().numpy().astype(np.uint8)
        img_path = data['path'][0]
        short_path = os.path.basename(img_path)
        name = os.path.splitext(short_path)[0]
        image_name = str(count)+'_%s.png' % name

        ks = np.array(list(range(opt.output_nc)))
        vs = ks * 255//len(ks)
        p = pred.copy()

        if is_save_image:
            # 创建一个新的彩色图像
            colored_mask = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)

            # 为每个类别分配颜色
            for class_value in range(opt.output_nc):
                if class_value == 0:
                    continue
                colored_mask[pred == class_value] = color_map[class_value]
            mask_dir = os.path.join(img_dir, 'mask')
            os.makedirs(mask_dir, exist_ok=True)
            # 保存彩色化的图像
            cv2.imwrite(os.path.join(mask_dir, image_name), colored_mask)

            # 创建一个新的彩色图像
            colored_mask = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)

            # 为每个类别分配颜色
            for class_value in range(opt.output_nc):
                if class_value == 0:
                    continue
                colored_mask[mask == class_value] = color_map[class_value]
            mask_dir = os.path.join(img_dir, 'ture')
            os.makedirs(mask_dir, exist_ok=True)
            # 保存彩色化的图像
            cv2.imwrite(os.path.join(mask_dir, image_name), colored_mask)
            # for k in ks:
            #     p[pred==k] = vs[k]
            # Image.fromarray(p.astype(np.uint8)).save(os.path.join(mask_dir, image_name))

        eval_start = time.time()


        print(f"bg_value: {bg_value}")
        print(f"pred: {pred.shape}")
        eval_result = get_total_evaluation(pred, mask, bg_value=bg_value)

        message = '%04d: %s \t' % (count, name)
        for k, v in eval_result.items():
            if k in eval_results:
                eval_results[k] += v
            else:
                eval_results[k] = v
            message += '%s: %.5f\t' % (k, v)

        eval_result['name'] = name
        result.append(eval_result)
        print(message, 'cost: %.4f' % (time.time() - eval_start))
        with open(img_dir + '_eval.txt', 'a') as log:
            log.write(message + '\n')

    message = 'total %d:\n' % count
    for k, v in eval_results.items():
        message += 'm_%s: %.5f\t' % (k, v / count)
    print(message)
    with open(img_dir + '_eval.txt', 'a') as log:
        log.write(message + '\n')
