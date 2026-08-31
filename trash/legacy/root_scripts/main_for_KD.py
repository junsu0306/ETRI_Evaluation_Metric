from args import get_parser
import os
import csv
import json
from os.path import join, exists, isfile
from os import makedirs
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import numpy as np
import math
from datetime import datetime
from tensorboardX import SummaryWriter
import h5py
import torchvision.models as models
import netvlad
import trainer
from trainer import train, test, get_clusters, save_checkpoint, Flatten, L2Norm, distillation_loss


if __name__ == "__main__":
    parser = get_parser()
    opt = parser.parse_args()

    CSV_FILE = f"./result_{opt.pruning_ratio}_{opt.nEpochs}.csv"
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as f:
            my_writer = csv.writer(f)
            my_writer.writerow(
                ["epoch", "recall@1", "recall@5", "recall@10", "recall@20"])

    restore_var = ['lr', 'lrStep', 'lrGamma', 'weightDecay', 'momentum',
                   'runsPath', 'savePath', 'arch', 'num_clusters', 'pooling', 'optim',
                   'margin', 'seed', 'patience']
    if opt.resume:
        flag_file = join(opt.resume, 'checkpoints', 'flags.json')
        if exists(flag_file):
            with open(flag_file, 'r') as f:
                stored_flags = {
                    '--'+k: str(v) for k, v in json.load(f).items() if k in restore_var}
                to_del = []
                for flag, val in stored_flags.items():
                    for act in parser._actions:
                        if act.dest == flag[2:]:
                            # store_true / store_false args don't accept arguments, filter these
                            if type(act.const) == type(True):
                                if val == str(act.default):
                                    to_del.append(flag)
                                else:
                                    stored_flags[flag] = ''
                for flag in to_del:
                    del stored_flags[flag]

                train_flags = [x for x in list(
                    sum(stored_flags.items(), tuple())) if len(x) > 0]
                print('Restored flags:', train_flags)
                opt = parser.parse_args(train_flags, namespace=opt)

    if opt.dataset.lower() == 'pittsburgh':
        import pittsburgh as dataset
    else:
        raise Exception('Unknown dataset')

    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")

    # set device number
    device = torch.device("cuda:"+opt.deviceNum if cuda else "cpu")

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    print('===> Loading dataset(s)')
    if opt.mode.lower() == 'train':
        whole_train_set = dataset.get_whole_training_set()
        whole_training_data_loader = DataLoader(dataset=whole_train_set,
                                                num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False,
                                                pin_memory=cuda)

        train_set = dataset.get_training_query_set(opt.margin)

        print('====> Training query set:', len(train_set))
        whole_test_set = dataset.get_whole_val_set()
        print('===> Evaluating on val set, query count:',
              whole_test_set.dbStruct.numQ)
    elif opt.mode.lower() == 'test':
        if opt.split.lower() == 'test':
            whole_test_set = dataset.get_whole_test_set()
            print('===> Evaluating on test set')
        elif opt.split.lower() == 'test250k':
            whole_test_set = dataset.get_250k_test_set()
            print('===> Evaluating on test250k set')
        elif opt.split.lower() == 'train':
            whole_test_set = dataset.get_whole_training_set()
            print('===> Evaluating on train set')
        elif opt.split.lower() == 'val':
            whole_test_set = dataset.get_whole_val_set()
            print('===> Evaluating on val set')
        else:
            raise ValueError('Unknown dataset split: ' + opt.split)
        print('====> Query count:', whole_test_set.dbStruct.numQ)
    elif opt.mode.lower() == 'cluster':
        whole_train_set = dataset.get_whole_training_set(onlyDB=True)

    print('===> Building model')

    pretrained = not opt.fromscratch
    if opt.arch.lower() == 'alexnet':
        encoder_dim = 256
        encoder = models.alexnet(pretrained=pretrained)
        # capture only features and remove last relu and maxpool
        layers = list(encoder.features.children())[:-2]

        if pretrained:
            # if using pretrained only train conv5
            for l in layers[:-1]:
                for p in l.parameters():
                    p.requires_grad = False

    elif opt.arch.lower() == 'vgg16':
        encoder_dim = 512
        encoder = models.vgg16(pretrained=pretrained)
        # capture only feature part and remove last relu and maxpool
        layers = list(encoder.features.children())[:-2]

        if pretrained:
            # if using pretrained then only train conv5_1, conv5_2, and conv5_3
            for l in layers[:-5]:
                for p in l.parameters():
                    p.requires_grad = False

    elif opt.arch.lower() == 'resnet-18':
        encoder_dim = 256
        encoder = models.resnet18(pretrained=True)
        # capture only feature part and remove last pool and fc

        layers = list(encoder.children())[:-3]
        pruning_block = layers[-1]

        if pretrained:
            for l in layers[:-1]:
                for p in l.parameters():
                    p.requires_grad = False

    elif opt.arch.lower() == 'mobilenetv2':  # we want to pruning this model

        model = models.mobilenet_v2(pretrained=True)
        encoder_dim = 320
        encoder = model.features
        layers = list(encoder.children())[:-1]

        if pretrained:
            for block in layers[:-1]:
                for p in block.parameters():
                    p.requires_grad = False

    elif opt.arch.lower() == 'mobilenetv3':

        model = models.mobilenet_v3_small(pretrained=True)
        encoder_dim = 96
        encoder = model.features
        layers = list(encoder.children())[:-1]

        if pretrained:
            for block in layers[:-1]:
                for p in block.parameters():
                    p.requires_grad = False

    if opt.mode.lower() == 'cluster' and not opt.vladv2:
        layers.append(L2Norm())

    encoder = nn.Sequential(*layers)
    model = nn.Module()
    model.add_module('encoder', encoder)

    # =================== generate teacher model =========================

    if opt.teacher_arch.lower() == 'resnet-18':
        teacher_encoder_dim = 256
        teacher_encoder = models.resnet18(pretrained=True)
        teacher_layers = list(teacher_encoder.children())[:-3]

        for l in teacher_layers:
            for p in l.parameters():
                p.requires_grad = False

    elif opt.teacher_arch.lower() == 'vgg16':
        teacher_encoder_dim = 512
        teacher_encoder = models.vgg16(pretrained=pretrained)
        # capture only feature part and remove last relu and maxpool
        teacher_layers = list(teacher_encoder.features.children())[:-2]

        for l in teacher_layers:
            for p in l.parameters():
                p.requires_grad = False

    else:
        raise ValueError('Unknown teacher architecture: ' + opt.teacher_arch)

    teacher_encoder = nn.Sequential(*teacher_layers)
    teacher_model = nn.Module()
    teacher_model.add_module('encoder', teacher_encoder)
    # ====================================================================

    if opt.mode.lower() != 'cluster':
        if opt.pooling.lower() == 'netvlad':
            net_vlad = netvlad.NetVLAD(
                num_clusters=opt.num_clusters, dim=encoder_dim, vladv2=opt.vladv2)
            taecher_net_vlad = netvlad.NetVLAD(
                num_clusters=opt.num_clusters, dim=teacher_encoder_dim, vladv2=opt.vladv2)

            if not opt.resume:
                if opt.mode.lower() == 'train':
                    initcache = join(opt.dataPath, 'centroids', opt.arch + '_' +
                                     train_set.dataset + '_' + str(opt.num_clusters) + '_desc_cen.hdf5')
                    # ============
                    init_teacher_cache = join(opt.dataPath, 'centroids', opt.teacher_arch + '_' +
                                              train_set.dataset + '_' + '64' + '_desc_cen.hdf5')  # clustering with vgg16
                    # ============
                else:
                    initcache = join(opt.dataPath, 'centroids', opt.arch + '_' +
                                     whole_test_set.dataset + '_' + str(opt.num_clusters) + '_desc_cen.hdf5')

                if not exists(initcache):
                    raise FileNotFoundError(
                        'Could not find clusters, please run with --mode=cluster before proceeding')

                with h5py.File(initcache, mode='r') as h5:
                    clsts = h5.get("centroids")[...]
                    traindescs = h5.get("descriptors")[...]
                    net_vlad.init_params(clsts, traindescs)
                    del clsts, traindescs

                with h5py.File(init_teacher_cache, mode='r') as h5:
                    teacher_clsts = h5.get("centroids")[...]
                    teacher_traindescs = h5.get("descriptors")[...]
                    taecher_net_vlad.init_params(
                        teacher_clsts, teacher_traindescs)
                    del teacher_clsts, teacher_traindescs

            teacher_model.add_module('pool', taecher_net_vlad)
            teacher_model.eval()

            model.add_module('pool', net_vlad)

        elif opt.pooling.lower() == 'max':
            global_pool = nn.AdaptiveMaxPool2d((1, 1))
            model.add_module('pool', nn.Sequential(
                *[global_pool, Flatten(), L2Norm()]))
        elif opt.pooling.lower() == 'avg':
            global_pool = nn.AdaptiveAvgPool2d((1, 1))
            model.add_module('pool', nn.Sequential(
                *[global_pool, Flatten(), L2Norm()]))
        else:
            raise ValueError('Unknown pooling type: ' + opt.pooling)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)

    print('===> Model:', opt.arch, 'with', opt.pooling, 'pooling')

    print('===> Total parameters:', total_params,
          'Trainable parameters:', trainable_params)

    # =======load teacher model =============

    if opt.mode.lower() == 'train':
        if opt.teacher_arch.lower() == 'resnet-18':
            teacher_resume_ckpt = join('./teacher/teacher_resnet-18_netvlad',
                                       'checkpoints', 'checkpoint.pth.tar')  # for teacher pretrained path

        elif opt.teacher_arch.lower() == 'vgg16':
            # for teacher pretrained path
            teacher_resume_ckpt = join(
                './teacher/teacher_vgg16_netvlad', 'checkpoints', 'checkpoint.pth.tar')
        else:

            raise ValueError(
                'Unknown teacher architecture: ' + opt.teacher_arch)

        if isfile(teacher_resume_ckpt):
            print("=> loading checkpoint '{}'".format(teacher_resume_ckpt))
            checkpoint = torch.load(teacher_resume_ckpt,
                                    map_location=lambda storage, loc: storage)

            teacher_model.load_state_dict(checkpoint['state_dict'])
            teacher_model = teacher_model.to(device)

            for l in teacher_model.parameters():
                l.requires_grad = False

            total_params = sum(p.numel() for p in teacher_model.parameters())
            trainable_params = sum(p.numel()
                                   for p in teacher_model.parameters() if p.requires_grad)

            print('=====> teacher model total paramerter: ', total_params)
            print('=====> teacher model trainable paramerter: ', trainable_params)

    # ====================================================================

    # =========== for adaptive pooling ===================
    cluster_size = opt.num_clusters
    Channel_t = teacher_encoder_dim
    Channel_s = encoder_dim
    spatial_pool = nn.AvgPool2d(kernel_size=2, stride=2)
    channel_gcd = math.gcd(Channel_t, Channel_s)
    # ====================================================

    isParallel = False

    if opt.nGPU > 1 and torch.cuda.device_count() > 1:
        model.encoder = nn.DataParallel(model.encoder)
        if opt.mode.lower() != 'cluster':
            model.pool = nn.DataParallel(model.pool)
        isParallel = True

    if not opt.resume:
        model = model.to(device)

    if opt.mode.lower() == 'train':
        if opt.optim.upper() == 'ADAM':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                          model.parameters()), lr=opt.lr)  # , betas=(0,0.9))
        elif opt.optim.upper() == 'SGD':
            optimizer = optim.SGD(
                list(filter(lambda p: p.requires_grad, model.parameters())),
                lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weightDecay
            )
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=opt.lrStep, gamma=opt.lrGamma)
        else:
            raise ValueError('Unknown optimizer: ' + opt.optim)

        # original paper/code doesn't sqrt() the distances, we do, so sqrt() the margin, I think :D
        criterion = nn.TripletMarginLoss(margin=opt.margin**0.5,
                                         p=2, reduction='sum').to(device)

    if opt.resume:
        if opt.ckpt.lower() == 'latest':
            resume_ckpt = join(opt.resume, 'checkpoints', 'checkpoint.pth.tar')
        elif opt.ckpt.lower() == 'best':
            resume_ckpt = join(opt.resume, 'checkpoints', 'model_best.pth.tar')

        if isfile(resume_ckpt):
            print("=> loading checkpoint '{}'".format(resume_ckpt))
            checkpoint = torch.load(
                resume_ckpt, map_location=lambda storage, loc: storage)
            opt.start_epoch = checkpoint['epoch']
            best_metric = checkpoint['best_score']
            model.load_state_dict(checkpoint['state_dict'])
            model = model.to(device)
            if opt.mode == 'train':
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_ckpt, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_ckpt))

    if opt.mode.lower() == 'test':
        print('===> Running evaluation step')
        model = torch.load("reduced_model.pth")
        model = model.to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)

        total_zero = 0
        trainable_zero = 0
        with torch.no_grad():
            for p in model.parameters():
                # 정확히 0인 요소 개수
                z = (p == 0).sum().item()
                total_zero += z

        print(f"Total parameters: {total_params}")
        print(f"Train parameters: {trainable_params}")
        print(f"zero parameters: {total_zero}")

        # trainer 모듈의 전역 변수들에 값 할당
        trainer.opt = opt
        trainer.model = model
        trainer.device = device
        trainer.cuda = cuda
        trainer.encoder_dim = encoder_dim
        trainer.CSV_FILE = f"./test_result_{opt.pruning_ratio}_{opt.nEpochs}.csv"

        epoch = 1
        recalls = test(whole_test_set, epoch, write_tboard=False)
    elif opt.mode.lower() == 'cluster':
        print('===> Calculating descriptors and clusters')

        # trainer 모듈의 전역 변수들에 값 할당
        trainer.opt = opt
        trainer.model = model
        trainer.device = device
        trainer.cuda = cuda
        trainer.encoder_dim = encoder_dim

        get_clusters(whole_train_set)
    elif opt.mode.lower() == 'train':
        print('===> Training model')
        writer = SummaryWriter(log_dir=join(opt.runsPath, datetime.now().strftime(
            '%b%d_%H-%M-%S')+'_'+opt.arch+'_'+opt.pooling))

        # write checkpoints in logdir
        logdir = writer.file_writer.get_logdir()
        opt.savePath = join(logdir, opt.savePath)
        if not opt.resume:
            makedirs(opt.savePath)

        with open(join(opt.savePath, 'flags.json'), 'w') as f:
            f.write(json.dumps(
                {k: v for k, v in vars(opt).items()}
            ))
        print('===> Saving state to:', logdir)

        # trainer 모듈의 전역 변수들에 값 할당
        trainer.opt = opt
        trainer.model = model
        trainer.teacher_model = teacher_model
        trainer.device = device
        trainer.cuda = cuda
        trainer.encoder_dim = encoder_dim
        trainer.whole_train_set = whole_train_set
        trainer.whole_training_data_loader = whole_training_data_loader
        trainer.train_set = train_set
        trainer.dataset = dataset
        trainer.criterion = criterion
        trainer.optimizer = optimizer
        trainer.writer = writer
        trainer.CSV_FILE = CSV_FILE

        not_improved = 0
        best_score = 0
        if opt.pruning == True:
            print(f"pruning variable is true")
        for epoch in range(opt.start_epoch+1, opt.nEpochs + 1):
            if opt.optim.upper() == 'SGD':
                scheduler.step(epoch)
            train(model, epoch)
            if (epoch % opt.evalEvery) == 0:
                recalls = test(whole_test_set, epoch, write_tboard=True)
                is_best = recalls[5] > best_score
                if is_best:
                    not_improved = 0
                    best_score = recalls[5]
                else:
                    not_improved += 1

                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'recalls': recalls,
                    'best_score': best_score,
                    'optimizer': optimizer.state_dict(),
                    'parallel': isParallel,
                }, is_best)

                if opt.patience > 0 and not_improved > (opt.patience / opt.evalEvery):
                    print('Performance did not improve for',
                          opt.patience, 'epochs. Stopping.')
                    break

        print("=> Best Recall@5: {:.4f}".format(best_score), flush=True)
        total_params = sum(p.numel() for p in model.parameters())

        nonzero_params = sum((p != 0).sum().item() for p in model.parameters())
        print(f"Total parameters: {total_params}")
        print(f"Non-zero parameters: {nonzero_params}")

        writer.close()
