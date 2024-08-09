from torch.utils.data import DataLoader
# from dataloader_for_RECOG import MyDataset
from .dataloader_for_DETECT import CenternetDataset

# -------------------------------------#
#       对数据集进行训练
# -------------------------------------#
import datetime
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from .FusionNet.nets.centernet import CenterNet_HourglassNet, CenterNet_Resnet50
from .FusionNet.nets.centernet_training import get_lr_scheduler, set_optimizer_lr
from .FusionNet._utils.callbacks import EvalCallback, LossHistory
from .FusionNet._utils.dataloader import CenternetDataset, centernet_dataset_collate
from .FusionNet._utils.utils import download_weights, get_classes, show_config
from .FusionNet._utils.utils_fit import fit_one_epoch
import scipy.io as scio

'''
训练自己的目标检测模型一定需要注意以下几点：
1、训练前仔细检查自己的格式是否满足要求，该库要求数据集格式为VOC格式，需要准备好的内容有输入图片和标签
   输入图片为.jpg图片，无需固定大小，传入训练前会自动进行resize。
   灰度图会自动转成RGB图片进行训练，无需自己修改。
   输入图片如果后缀非jpg，需要自己批量转成jpg后再开始训练。

   标签为.xml格式，文件中会有需要检测的目标信息，标签文件和输入图片文件相对应。

2、损失值的大小用于判断是否收敛，比较重要的是有收敛的趋势，即验证集损失不断下降，如果验证集损失基本上不改变的话，模型基本上就收敛了。
   损失值的具体大小并没有什么意义，大和小只在于损失的计算方式，并不是接近于0才好。如果想要让损失好看点，可以直接到对应的损失函数里面除上10000。
   训练过程中的损失值会保存在logs文件夹下的loss_%Y_%m_%d_%H_%M_%S文件夹中

3、训练好的权值文件保存在logs文件夹中，每个训练世代（Epoch）包含若干训练步长（Step），每个训练步长（Step）进行一次梯度下降。
   如果只是训练了几个Step是不会保存的，Epoch和Step的概念要捋清楚一下。
'''
###########################################################
# 目标检测：双模训练
###########################################################
def train_mm(image_path='', hrrp_path='', label_path='', class_name=None,
             num_classes=2, epochs=100, batch_size=8, learning_rate=5e-4, loadPretrain=0):
    Cuda = True

    if class_name is None:
        class_name = []

    if not os.path.exists('./weights'):
        os.makedirs('./weights')

    print('开始进行红外/雷达的双模态目标检测训练...')

    # 后期修改voc_annotation.py
    train_annotation_path = './2007_train.txt'
    hrrp_path = hrrp_path

    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()

    with open(label_path, 'r') as f:
        next(f)
        label_lines = f.readlines()

    input_shape = [512, 512]

    # train_dataset = MyDataset(image_path=image_path, hrrp_path=hrrp_path, label_path=label_path, class_name=class_name)
    sync_bn = False
    distributed = False

    # ---------------------------------------------------------------------#
    #   fp16        是否使用混合精度训练
    #               可减少约一半的显存、需要pytorch1.7.1以上
    # ---------------------------------------------------------------------#
    fp16 = True
    # ---------------------------------------------------------------------#
    #   classes_path    指向model_data下的txt，与自己训练的数据集相关
    #                   训练前一定要修改classes_path，使其对应自己的数据集
    # ---------------------------------------------------------------------#
    classes_path = './model_data/my_classes.txt'

    model_path = ''

    backbone = "resnet50"

    pretrained = False

    Init_Epoch = 0
    Freeze_Epoch = 50
    Freeze_batch_size = 16

    UnFreeze_Epoch = epochs
    Unfreeze_batch_size = 8

    Freeze_Train = True

    Init_lr = 5e-4
    Min_lr = Init_lr * 0.01
    optimizer_type = "adam"
    momentum = 0.9
    weight_decay = 0
    lr_decay_type = 'cos'
    save_period = 5
    save_dir = 'logs'

    eval_flag = True
    eval_period = 1

    num_workers = 4

    train_annotation_path = './2007_train.txt'
    val_annotation_path = './2007_val.txt'

    # ------------------------------------------------------#
    #   设置用到的显卡
    # ------------------------------------------------------#
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0

        # ----------------------------------------------------#
        #   下载预训练权重
        # ----------------------------------------------------#
        if pretrained:
            if distributed:
                if local_rank == 0:
                    download_weights(backbone)
                dist.barrier()
            else:
                download_weights(backbone)

        # ----------------------------------------------------#
        #   获取classes
        # ----------------------------------------------------#
        class_names, num_classes = get_classes(classes_path)

        if backbone == "resnet50":
            model = CenterNet_Resnet50(num_classes, pretrained=pretrained)
            with open('model_data/pth_name.txt', 'r') as f:
                model_path = f.readline()
                f.close()
            model_path = os.path.join('weights/multimodel',model_path)

            model_dict = model.state_dict()
            pretrained_dict = torch.load(model_path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict
                               and v.size() == model_dict[k].size()}

            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            # model.load_state_dict(torch.load(model_path, map_location=device),strict=False)
        else:
            model = CenterNet_HourglassNet({'hm': num_classes, 'wh': 2, 'reg': 2}, pretrained=pretrained)
        if model_path != '':
            # ------------------------------------------------------#
            #   权值文件请看README，百度网盘下载
            # ------------------------------------------------------#
            if local_rank == 0:
                print('Load weights {}.'.format(model_path))

            # ------------------------------------------------------#
            #   根据预训练权重的Key和模型的Key进行加载
            # ------------------------------------------------------#
            model_dict = model.state_dict()
            pretrained_dict = torch.load(model_path, map_location=device)
            load_key, no_load_key, temp_dict = [], [], {}
            for k, v in pretrained_dict.items():
                if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                    temp_dict[k] = v
                    load_key.append(k)
                else:
                    no_load_key.append(k)
            model_dict.update(temp_dict)
            model.load_state_dict(model_dict)
            # ------------------------------------------------------#
            #   显示没有匹配上的Key
            # ------------------------------------------------------#
            if local_rank == 0:
                print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
                print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
                print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

        # ----------------------#
        #   记录Loss
        # ----------------------#
        if local_rank == 0:
            time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
            log_dir = os.path.join(save_dir, "loss_" + str(time_str))
            loss_history = LossHistory(log_dir, model, input_shape=input_shape)
        else:
            loss_history = None

        # ------------------------------------------------------------------#
        #   torch 1.2不支持amp，建议使用torch 1.7.1及以上正确使用fp16
        #   因此torch1.2这里显示"could not be resolve"
        # ------------------------------------------------------------------#
        if fp16:
            from torch.cuda.amp import GradScaler as GradScaler

            scaler = GradScaler()
        else:
            scaler = None

        model_train = model.train()
        # ----------------------------#
        #   多卡同步Bn
        # ----------------------------#
        if sync_bn and ngpus_per_node > 1 and distributed:
            model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
        elif sync_bn:
            print("Sync_bn is not support in one gpu or not distributed.")

        if Cuda:
            if distributed:
                # ----------------------------#
                #   多卡平行运行
                # ----------------------------#
                model_train = model_train.cuda(local_rank)
                model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank],
                                                                        find_unused_parameters=True)
            else:
                model_train = torch.nn.DataParallel(model)
                cudnn.benchmark = True
                model_train = model_train.cuda()

        # ---------------------------#
        #   读取数据集对应的txt
        # ---------------------------#
        with open(train_annotation_path, encoding='utf-8') as f:
            train_lines = f.readlines()
        with open(val_annotation_path, encoding='utf-8') as f:
            val_lines = f.readlines()
        num_train = len(train_lines)
        num_val = len(val_lines)

        if local_rank == 0:
            show_config(
                classes_path=classes_path, model_path=model_path, input_shape=input_shape, \
                Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
                Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size, Freeze_Train=Freeze_Train, \
                Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum,
                lr_decay_type=lr_decay_type, \
                save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train,
                num_val=num_val
            )
            # ---------------------------------------------------------#
            #   总训练世代指的是遍历全部数据的总次数
            #   总训练步长指的是梯度下降的总次数
            #   每个训练世代包含若干训练步长，每个训练步长进行一次梯度下降。
            #   此处仅建议最低训练世代，上不封顶，计算时只考虑了解冻部分
            # ----------------------------------------------------------#
            wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
            total_step = num_train // Unfreeze_batch_size * UnFreeze_Epoch
            if total_step <= wanted_step:
                if num_train // Unfreeze_batch_size == 0:
                    raise ValueError('数据集过小，无法进行训练，请扩充数据集。')
                wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
                print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m" % (optimizer_type, wanted_step))
                print("\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，Unfreeze_batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m" % (
                    num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
                print("\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m" % (
                    total_step, wanted_step, wanted_epoch))

        # ------------------------------------------------------#
        #   主干特征提取网络特征通用，冻结训练可以加快训练速度
        #   也可以在训练初期防止权值被破坏。
        #   Init_Epoch为起始世代
        #   Freeze_Epoch为冻结训练的世代
        #   UnFreeze_Epoch总训练世代
        #   提示OOM或者显存不足请调小Batch_size
        # ------------------------------------------------------#
        if True:
            UnFreeze_flag = False
            # ------------------------------------#
            #   冻结一定部分训练
            # ------------------------------------#
            if Freeze_Train:
                model.freeze_backbone()

            # -------------------------------------------------------------------#
            #   如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
            # -------------------------------------------------------------------#
            batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

            # -------------------------------------------------------------------#
            #   判断当前batch_size，自适应调整学习率
            # -------------------------------------------------------------------#
            nbs = 64
            lr_limit_max = 5e-4 if optimizer_type == 'adam' else 5e-2
            lr_limit_min = 2.5e-4 if optimizer_type == 'adam' else 5e-4
            Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
            Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

            # ---------------------------------------#
            #   根据optimizer_type选择优化器
            # ---------------------------------------#
            optimizer = {
                'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
                'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True,
                                 weight_decay=weight_decay)
            }[optimizer_type]

            # ---------------------------------------#
            #   获得学习率下降的公式
            # ---------------------------------------#
            lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

            # ---------------------------------------#
            #   判断每一个世代的长度
            # ---------------------------------------#
            epoch_step = num_train // batch_size
            epoch_step_val = num_val // batch_size

            if epoch_step == 0 or epoch_step_val == 0:
                raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

            try:
                hrrp_lines = scio.loadmat(hrrp_path)['aa']
            except Exception as e:
                hrrp_lines = scio.loadmat(hrrp_path)['raw_data']

            with open(label_path, 'r') as f:
                next(f)
                label_lines = f.readlines()

            train_dataset = CenternetDataset(train_lines, hrrp_lines, label_lines, input_shape, num_classes,
                                             train=True)
            val_dataset = CenternetDataset(val_lines, hrrp_lines, label_lines, input_shape, num_classes, train=False)

            if distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, )
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=True, )
                batch_size = batch_size // ngpus_per_node
                shuffle = False
            else:
                train_sampler = None
                val_sampler = None
                shuffle = True

            gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True, collate_fn=centernet_dataset_collate, sampler=train_sampler)
            gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True, collate_fn=centernet_dataset_collate, sampler=val_sampler)

            # ----------------------#
            #   记录eval的map曲线
            # ----------------------#
            if local_rank == 0:
                eval_callback = EvalCallback(model, backbone, input_shape, class_names, num_classes, val_lines,
                                             hrrp_lines, label_lines, log_dir,
                                             Cuda, \
                                             eval_flag=eval_flag, period=eval_period)
            else:
                eval_callback = None

            # ---------------------------------------#
            #   开始模型训练
            # ---------------------------------------#
            for epoch in range(Init_Epoch, UnFreeze_Epoch):
                with open('model_data/sigal.event', 'r') as f:
                    sig = int(f.read())
                    f.close()
                if sig==1:
                    print('停止训练！！！！')
                    break
                # ---------------------------------------#
                #   如果模型有冻结学习部分
                #   则解冻，并设置参数
                # ---------------------------------------#
                if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                    batch_size = Unfreeze_batch_size

                    # -------------------------------------------------------------------#
                    #   判断当前batch_size，自适应调整学习率
                    # -------------------------------------------------------------------#
                    nbs = 64
                    lr_limit_max = 5e-4 if optimizer_type == 'adam' else 5e-2
                    lr_limit_min = 2.5e-4 if optimizer_type == 'adam' else 5e-4
                    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                    # ---------------------------------------#
                    #   获得学习率下降的公式
                    # ---------------------------------------#
                    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                    model.unfreeze_backbone()

                    epoch_step = num_train // batch_size
                    epoch_step_val = num_val // batch_size

                    if epoch_step == 0 or epoch_step_val == 0:
                        raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                    if distributed:
                        batch_size = batch_size // ngpus_per_node

                    gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=True,
                                     drop_last=True, collate_fn=centernet_dataset_collate, sampler=train_sampler)

                    gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                         pin_memory=True,
                                         drop_last=True, collate_fn=centernet_dataset_collate, sampler=val_sampler)

                    UnFreeze_flag = True

                if distributed:
                    train_sampler.set_epoch(epoch)

                set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

                fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch,
                              epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, fp16, scaler, backbone,
                              save_period, save_dir, local_rank)

            if local_rank == 0:
                loss_history.writer.close()

    print(f'双模态训练完成！权重文件已保存至./weights目录下!')
    print('请继续操作！')


if __name__ == '__main__':

    train_mm(image_path='./test_data/car_building/VOCdevkit/VOC2007/JPEGImages',
             hrrp_path='./test_data/car_building/hrrps.mat',
             label_path='./test_data/car_building/label.csv',
                     class_name=['car','building'],
                     num_classes=2, epochs=10, batch_size=8, learning_rate=0.001)
