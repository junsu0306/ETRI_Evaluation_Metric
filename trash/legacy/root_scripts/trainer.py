from __future__ import print_function
from math import log10, ceil
import random
import shutil
import json
from os.path import join, exists, isfile, realpath, dirname
from os import makedirs, remove, chdir, environ
import os
import csv
from utils.pruning.common import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataset import Subset
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime
import torchvision.datasets as datasets
import torchvision.models as models
import h5py
import faiss
from tqdm.auto import tqdm
import math
from tensorboardX import SummaryWriter
import numpy as np


# 전역 변수 선언 - main_for_KD.py에서 정의되는 변수들
opt = None
model = None
teacher_model = None
device = None
cuda = None
encoder_dim = None
whole_train_set = None
whole_training_data_loader = None
train_set = None
dataset = None
criterion = None
optimizer = None
writer = None
CSV_FILE = None


def distillation_loss(student_logits, teacher_logits, T=4.0):

    # 1) 공간 차원(H,W) 평균
    s = student_logits.view(student_logits.size(
        0), student_logits.size(1), -1).mean(dim=2)  # -> [B, C]
    t = teacher_logits.view(teacher_logits.size(
        0), teacher_logits.size(1), -1).mean(dim=2)  # -> [B, C]

    # 2) KD loss
    log_p_s = F.log_softmax(s / T, dim=1)
    p_t = F.softmax(t / T, dim=1)
    kd = F.kl_div(log_p_s, p_t, reduction='batchmean') * (T * T)
    return kd


def train(model, epoch):
    epoch_loss = 0
    startIter = 1  # keep track of batch iter across subsets for logging

    if opt.cacheRefreshRate > 0:
        # only using get_training_query_set()
        subsetN = ceil(len(train_set) / opt.cacheRefreshRate)
        subsetIdx = np.array_split(np.arange(len(train_set)), subsetN)
    else:
        subsetN = 1
        subsetIdx = [np.arange(len(train_set))]

    nBatches = (len(train_set) + opt.batchSize - 1) // opt.batchSize

    for subIter in range(subsetN):

        print('====> Building Cache')
        model.eval()
        train_set.cache = join(
            opt.cachePath, train_set.whichSet + '_feat_cache.hdf5')
        with h5py.File(train_set.cache, mode='w') as h5:
            pool_size = encoder_dim
            if opt.pooling.lower() == 'netvlad':
                pool_size *= opt.num_clusters
            h5feat = h5.create_dataset("features",
                                       [len(whole_train_set), pool_size],
                                       dtype=np.float32)
            with torch.no_grad():
                for iteration, (input, indices) in enumerate(whole_training_data_loader, 1):
                    input = input.to(device)
                    image_encoding = model.encoder(input)
                    vlad_encoding = model.pool(image_encoding)
                    h5feat[indices.detach().numpy(),
                           :] = vlad_encoding.detach().cpu().numpy()
                    del input, image_encoding, vlad_encoding

        sub_train_set = Subset(dataset=train_set, indices=subsetIdx[subIter])

        training_data_loader = DataLoader(dataset=sub_train_set, num_workers=opt.threads,
                                          batch_size=opt.batchSize, shuffle=True,
                                          collate_fn=dataset.collate_fn, pin_memory=cuda)
#################
        pbar = tqdm(
            enumerate(training_data_loader, startIter),
            total=len(training_data_loader),
            unit="batch",
            desc=f"Epoch {epoch}/{opt.nEpochs}  |  subset {subIter+1}/{subsetN}",
            leave=False                       # for using tqdm
        )
################
        print('Allocated:', torch.cuda.memory_allocated())
        print('Cached:', torch.cuda.memory_cached())

        model.train()
        for iteration, (query, positives, negatives, negCounts, indices) in pbar:
            # some reshaping to put query, pos, negs in a single (N, 3, H, W) tensor
            # where N = batchSize * (nQuery + nPos + nNeg)
            if query is None:
                continue  # in case we get an empty batch

            B, C, H, W = query.shape
            nNeg = torch.sum(negCounts)
            input = torch.cat([query, positives, negatives])
            # ====================================

            input = input.to(device)

            image_encoding = model.encoder(input)
            vlad_encoding = model.pool(image_encoding)

            # ================== for distillation loss =========================
            with torch.no_grad():
                teacher_feat = teacher_model.encoder(input)
                teacher_soft_assign_logit = teacher_model.pool.get_soft_assign_logit(
                    teacher_feat)
                stu_soft_assign_logit = model.pool.get_soft_assign_logit(
                    image_encoding)
            # =====================================

            vladQ, vladP, vladN = torch.split(vlad_encoding, [B, B, nNeg])
            optimizer.zero_grad()

            # calculate loss for each Query, Positive, Negative triplet
            # due to potential difference in number of negatives have to
            # do it per query, per negative
            loss = 0
            for i, negCount in enumerate(negCounts):
                for n in range(negCount):
                    negIx = (torch.sum(negCounts[:i]) + n).item()
                    loss += criterion(vladQ[i:i+1],
                                      vladP[i:i+1], vladN[negIx:negIx+1])

            loss /= nNeg.float().to(device)
            distill_pram = 0.5
            distill_loss = distillation_loss(
                stu_soft_assign_logit, teacher_soft_assign_logit, T=1.5)

            total_loss = (1-distill_pram)*loss + distill_pram*distill_loss
            total_loss.backward()
            optimizer.step()

            #### start pruning ####
            # model = mobilenetv2_L_block_pruning(model, sparsity=0.15)

            # Assuming the last conv layer is at the end of the encoder
            last_conv_layer = model.encoder[-1].conv[-2].to(device)
            # Assuming the last layer is
            last_bn_layer = model.encoder[-1].conv[-1].to(device)

            #print(f"Last Conv Layer: {last_conv_layer}")
            #print(f"Last BatchNorm Layer: {last_bn_layer}")
            mobilenetv2_L_layer_pair_pruning(
                last_conv_layer, last_bn_layer, sparsity=0.15)

            # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            # nonzero_params = sum((p != 0).sum().item() for p in model.parameters())
            # zero_params = sum((p  == 0).sum().item() for p in model.parameters())

            # print(f"Trainable params: {trainable_params}, Non-zero params: {nonzero_params}, Zero params: {zero_params}")

            # #### end pruning ####

            writer.add_scalar('Train/DistillLoss', distill_loss.item(),
                              ((epoch-1) * nBatches) + iteration)

            # cleanup
            del distill_loss, teacher_feat
            del input, image_encoding, vlad_encoding, vladQ, vladP, vladN
            del query, positives, negatives

            batch_loss = loss.item()
            epoch_loss += batch_loss

            pbar.set_postfix(
                loss=f"{batch_loss:0.4f}",
                nNeg=int(nNeg),
                mem=f"{torch.cuda.memory_allocated()//1e6}MB"
            )

            if iteration % 50 == 0 or nBatches <= 10:
                print("==> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration,
                                                                  nBatches, batch_loss), flush=True)
                writer.add_scalar('Train/Loss', batch_loss,
                                  ((epoch-1) * nBatches) + iteration)
                writer.add_scalar('Train/nNeg', nNeg,
                                  ((epoch-1) * nBatches) + iteration)
                print('Allocated:', torch.cuda.memory_allocated())
                print('Cached:', torch.cuda.memory_cached())

        startIter += len(training_data_loader)
        del training_data_loader, loss
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        remove(train_set.cache)  # delete HDF5 cache

    avg_loss = epoch_loss / nBatches

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, avg_loss),
          flush=True)
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch)


def test(eval_set, epoch=0, write_tboard=False):
    # TODO what if features dont fit in memory?
    test_data_loader = DataLoader(dataset=eval_set,
                                  num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False,
                                  pin_memory=cuda)

    model.eval()
    with torch.no_grad():
        print('====> Extracting Features')
        # This is a simplified code. If an error occurs in this part, please check the model encoder dimension setting section.
        try:
            pool_size = model.encoder_dim
        except:
            pool_size = encoder_dim

        if opt.pooling.lower() == 'netvlad':
            pool_size *= opt.num_clusters
        dbFeat = np.empty((len(eval_set), pool_size))

        for iteration, (input, indices) in enumerate(test_data_loader, 1):
            input = input.to(device)
            image_encoding = model.encoder(input)
            vlad_encoding = model.pool(image_encoding)

            dbFeat[indices.detach().numpy(),
                   :] = vlad_encoding.detach().cpu().numpy()
            if iteration % 50 == 0 or len(test_data_loader) <= 10:
                print("==> Batch ({}/{})".format(iteration,
                                                 len(test_data_loader)), flush=True)

            del input, image_encoding, vlad_encoding
    del test_data_loader

    # extracted for both db and query, now split in own sets
    qFeat = dbFeat[eval_set.dbStruct.numDb:].astype('float32')
    dbFeat = dbFeat[:eval_set.dbStruct.numDb].astype('float32')

    print('====> Building faiss index')
    faiss_index = faiss.IndexFlatL2(pool_size)
    faiss_index.add(dbFeat)

    print('====> Calculating recall @ N')
    n_values = [1, 5, 10, 20]

    _, predictions = faiss_index.search(qFeat, max(n_values))

    # for each query get those within threshold distance
    gt = eval_set.getPositives()

    correct_at_n = np.zeros(len(n_values))
    # TODO can we do this on the matrix in one go?
    for qIx, pred in enumerate(predictions):
        for i, n in enumerate(n_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break
    recall_at_n = correct_at_n / eval_set.dbStruct.numQ

    recalls = {}  # make dict for output
    for i, n in enumerate(n_values):
        recalls[n] = recall_at_n[i]
        print("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))
        if write_tboard:
            writer.add_scalar('Val/Recall@' + str(n), recall_at_n[i], epoch)

    with open(CSV_FILE, "a", newline="") as f:
        my_writer = csv.writer(f)
        my_writer.writerow([
            epoch + 1,
            f"{recall_at_n[0]:.4f}",
            f"{recall_at_n[1]:.4f}",
            f"{recall_at_n[2]:.4f}",
            f"{recall_at_n[3]:.4f}"
        ])

    return recalls


def get_clusters(cluster_set):
    nDescriptors = 50000
    nPerImage = 100
    nIm = ceil(nDescriptors/nPerImage)

    sampler = SubsetRandomSampler(np.random.choice(
        len(cluster_set), nIm, replace=False))
    data_loader = DataLoader(dataset=cluster_set,
                             num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False,
                             pin_memory=cuda,
                             sampler=sampler)

    if not exists(join(opt.dataPath, 'centroids')):
        makedirs(join(opt.dataPath, 'centroids'))

    initcache = join(opt.dataPath, 'centroids', opt.arch + '_' +
                     cluster_set.dataset + '_' + str(opt.num_clusters) + '_desc_cen.hdf5')
    with h5py.File(initcache, mode='w') as h5:
        with torch.no_grad():
            model.eval()
            print('====> Extracting Descriptors')
            dbFeat = h5.create_dataset("descriptors",
                                       [nDescriptors, encoder_dim],
                                       dtype=np.float32)

            for iteration, (input, indices) in enumerate(data_loader, 1):
                input = input.to(device)
                image_descriptors = model.encoder(input).view(
                    input.size(0), encoder_dim, -1).permute(0, 2, 1)

                batchix = (iteration-1)*opt.cacheBatchSize*nPerImage
                for ix in range(image_descriptors.size(0)):
                    # sample different location for each image in batch
                    sample = np.random.choice(
                        image_descriptors.size(1), nPerImage, replace=False)
                    startix = batchix + ix*nPerImage
                    dbFeat[startix:startix+nPerImage, :] = image_descriptors[ix,
                                                                             sample, :].detach().cpu().numpy()

                if iteration % 50 == 0 or len(data_loader) <= 10:
                    print("==> Batch ({}/{})".format(iteration,
                                                     ceil(nIm/opt.cacheBatchSize)), flush=True)
                del input, image_descriptors

        print('====> Clustering..')
        niter = 100
        kmeans = faiss.Kmeans(encoder_dim, opt.num_clusters,
                              niter=niter, verbose=False)
        kmeans.train(dbFeat[...])

        print('====> Storing centroids', kmeans.centroids.shape)
        h5.create_dataset('centroids', data=kmeans.centroids)
        print('====> Done!')


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    model_out_path = join(opt.savePath, filename)
    print(f"model output path: {model_out_path}")
    torch.save(state, model_out_path)
    if is_best:
        shutil.copyfile(model_out_path, join(
            opt.savePath, 'model_best.pth.tar'))


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)
