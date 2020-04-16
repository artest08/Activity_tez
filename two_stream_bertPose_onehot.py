#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 18:37:08 2020

@author: esat
"""

import os
import time
import argparse
import shutil
import numpy as np


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from tensorboardX import SummaryWriter
from opt.AdamW import AdamW
from torch.optim import lr_scheduler

import video_transforms
import models
import datasets
import swats


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

dataset_names = sorted(name for name in datasets.__all__)

parser = argparse.ArgumentParser(description='PyTorch Two-Stream Action Recognition')
#parser.add_argument('--data', metavar='DIR', default='./datasets/ucf101_frames',
#                    help='path to dataset')
parser.add_argument('--settings', metavar='DIR', default='./datasets/settings',
                    help='path to datset setting files')
#parser.add_argument('--modality', '-m', metavar='MODALITY', default='rgb',
#                    choices=["rgb", "flow"],
#                    help='modality: rgb | flow')
parser.add_argument('--dataset', '-d', default='hmdb51',
                    choices=["ucf101", "hmdb51", "smtV2", "window"],
                    help='dataset: ucf101 | hmdb51')
parser.add_argument('--arch', '-a', metavar='ARCH', default='poseRaw2_bert7',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: rgb_vgg16)')
parser.add_argument('-s', '--split', default=1, type=int, metavar='S',
                    help='which split of data to work on (default: 1)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 50)')
parser.add_argument('--iter-size', default=1, type=int,
                    metavar='I', help='iter size as in Caffe to reduce memory usage (default: 5)')
parser.add_argument('--new_width', default=340, type=int,
                    metavar='N', help='resize width (default: 340)')
parser.add_argument('--new_height', default=256, type=int,
                    metavar='N', help='resize height (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--save-freq', default=1, type=int,
                    metavar='N', help='save frequency (default: 25)')
parser.add_argument('--num-seg', default=16, type=int,
                    metavar='N', help='Number of segments for temporal LSTM (default: 16)')
#parser.add_argument('--resume', default='./dene4', type=str, metavar='PATH',
#                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('-c', '--continue', dest='contine', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('-r', '--ranking', dest='ranking', action='store_true',
                    help='enable ranking mode')



best_prec1 = 0
best_loss = 30
mseCoeffStart=10



def main():
    global args, best_prec1,writer,best_loss,mseCoeffStart, length
    args = parser.parse_args()
    
    saveLocation="./checkpoint/"+args.dataset+"_"+args.arch+"_split"+str(args.split)
    if not os.path.exists(saveLocation):
        os.makedirs(saveLocation)
    writer = SummaryWriter(saveLocation)
   
    # create model

    if args.evaluate:
        print("Building validation model ... ")
        model = build_model_validate()
    elif args.contine:
        model, startEpoch, optimizer, best_prec1 = build_model_continue()
        lr = None
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        print("Continuing with best precision: %.3f and start epoch %d and lr: %f" %(best_prec1,startEpoch,lr))
    else:
        print("Building model ... ")
        model = build_model()
        #optimizer = torch.optim.Adam(model.parameters(), args.lr)
        optimizer = AdamW(model.parameters(), lr= args.lr, weight_decay=args.weight_decay)
        #optimizer = swats.SWATS(model.parameters(), args.lr)
        #model = build_model_validate()
        startEpoch = 0
    

    
    print("Model %s is loaded. " % (args.arch))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    criterion2 =  nn.BCEWithLogitsLoss().cuda()
    
    

#    optimizer = torch.optim.SGD(model.parameters(), args.lr,
#                                momentum=args.momentum,
#                                weight_decay=args.weight_decay)
    

    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', patience=300,verbose=True)    
    
    print("Saving everything to directory %s." % (saveLocation))
    if args.dataset=='ucf101':
        dataset='./datasets/pose_information2/ucf101'
    elif args.dataset=='hmdb51':
        dataset='./datasets/pose_information2/hmdb51'
    elif args.dataset=='smtV2':
        dataset='./datasets/pose_information2/smtV2'
    elif args.dataset=='window':
        dataset='./datasets/pose_information2/window'
    else:
        print("No convenient dataset entered, exiting....")
        return 0
    
    cudnn.benchmark = True
    modality=args.arch.split('_')[0]
    length=1
    # Data transforming

    scale_ratios = [1.0, 0.875, 0.75, 0.66]



    train_transform = video_transforms.Compose([
            video_transforms.rawPoseAugmentation(scale_ratios),
            video_transforms.pose_one_hot_decoding2(args.num_seg),
        ])



    val_transform = video_transforms.Compose([
            video_transforms.rawPoseAugmentation([1.0]),
            video_transforms.pose_one_hot_decoding2(args.num_seg),
        ])
    validation_batch_size = int(args.batch_size)
    # data loading

    train_setting_file = "train_%s_split%d.txt" % ('rgb', args.split)
    train_split_file = os.path.join(args.settings, args.dataset, train_setting_file)
    val_setting_file = "val_%s_split%d.txt" % ('rgb', args.split)
    val_split_file = os.path.join(args.settings, args.dataset, val_setting_file)
    if not os.path.exists(train_split_file) or not os.path.exists(val_split_file):
        print("No split file exists in %s directory. Preprocess the dataset first" % (args.settings))

    train_dataset = datasets.__dict__[args.dataset](root=dataset,
                                                    source=train_split_file,
                                                    phase="train",
                                                    modality=modality,
                                                    is_color=False,
                                                    new_length=length,
                                                    new_width=args.new_width,
                                                    new_height=args.new_height,
                                                    video_transform=train_transform,
                                                    num_segments=args.num_seg)
    
    val_dataset = datasets.__dict__[args.dataset](root=dataset,
                                                  source=val_split_file,
                                                  phase="val",
                                                  modality=modality,
                                                  is_color=False,
                                                  new_length=length,
                                                  new_width=args.new_width,
                                                  new_height=args.new_height,
                                                  video_transform=val_transform,
                                                  num_segments=args.num_seg)

    print('{} samples found, {} train samples and {} test samples.'.format(len(val_dataset)+len(train_dataset),
                                                                           len(train_dataset),
                                                                           len(val_dataset)))
    if torch.cuda.device_count() > 1:
        drop_last_value = True
    else:
        drop_last_value = False
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last = drop_last_value)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size = validation_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last = drop_last_value)

    if args.evaluate:
        prec1,prec3=validate(val_loader, model, criterion)
        return

    for epoch in range(startEpoch, args.epochs):
        #adjust_learning_rate(optimizer, epoch)
        setMseCoeff=adjust_mse_coeff(mseCoeffStart, epoch)
        # train for one epoch
        train(train_loader, model, criterion,criterion2, optimizer, epoch,setMseCoeff,modality)

        # evaluate on validation set
        prec1 = 0.0
        lossClassification = 0
        lossMSE = None
        if (epoch + 1) % args.save_freq == 0:
            prec1, prec3, lossClassification, lossMSE = validate(val_loader, model, criterion,criterion2,modality)
            writer.add_scalar('data/top1_validation', prec1, epoch)
            writer.add_scalar('data/top3_validation', prec3, epoch)
            writer.add_scalar('data/classification_loss_validation', lossClassification, epoch)
            scheduler.step(prec1)
        # remember best prec@1 and save checkpoint
        
        is_best = prec1 >= best_prec1
        best_prec1 = max(prec1, best_prec1)
        # if lossMSE == None:
        #     is_best = True
        #     best_loss = lossMSE
        # else:
        #     is_best = lossMSE < best_loss
        #     best_loss = min(lossMSE, best_loss)

        if (epoch + 1) % args.save_freq == 0:
            checkpoint_name = "%03d_%s" % (epoch + 1, "checkpoint.pth.tar")
            if is_best:
                print("Model son iyi olarak kaydedildi")
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'best_loss': best_loss,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, checkpoint_name, saveLocation)
    
    checkpoint_name = "%03d_%s" % (epoch + 1, "checkpoint.pth.tar")
    save_checkpoint({
        'epoch': epoch + 1,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'best_loss': best_loss,
        'optimizer' : optimizer.state_dict(),
    }, is_best, checkpoint_name, saveLocation)
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()

def build_model():
    modelLocation="./checkpoint/"+args.dataset+"_"+'_'.join(args.arch.split('_')[:-1])+"_split"+str(args.split)
    modality=args.arch.split('_')[0]

    model_path='' 
    if args.dataset=='ucf101':
        print('model path is: %s' %(model_path))
        model = models.__dict__[args.arch](modelPath=model_path, num_classes=101,length=args.num_seg)
    elif args.dataset=='hmdb51':
        print('model path is: %s' %(model_path))
        model = models.__dict__[args.arch](modelPath=model_path, num_classes=51, length=args.num_seg)
    elif args.dataset=='smtV2':
        print('model path is: %s' %(model_path))
        model = models.__dict__[args.arch](modelPath=model_path, num_classes=174, length=args.num_seg)  
    elif args.dataset=='window':
        print('model path is: %s' %(model_path))
        model = models.__dict__[args.arch](modelPath=model_path, num_classes=3, length=args.num_seg)  
        
    if torch.cuda.device_count() > 1:
        print('Multi-GPU test enabled...')
        model=torch.nn.DataParallel(model)
    #model.load_state_dict(torch.load('./weights/pose_pretrain.pth')['state_dict'])
    model = model.cuda()
    
    return model

def build_model_validate():
    modelLocation="./checkpoint/"+args.dataset+"_"+args.arch+"_split"+str(args.split)
    model_path = os.path.join(modelLocation,'model_best.pth.tar') 
    params = torch.load(model_path)
    print(modelLocation)
    if args.dataset=='ucf101':
        print('model path is: %s' %(model_path))
        model = models.__dict__[args.arch](modelPath=model_path, num_classes=101,length=args.num_seg)
    elif args.dataset=='hmdb51':
        print('model path is: %s' %(model_path))
        model = models.__dict__[args.arch](modelPath=model_path, num_classes=51, length=args.num_seg)
    elif args.dataset=='smtV2':
        print('model path is: %s' %(model_path))
        model = models.__dict__[args.arch](modelPath=model_path, num_classes=174, length=args.num_seg)
    elif args.dataset=='window':
        print('model path is: %s' %(model_path))
        model = models.__dict__[args.arch](modelPath=model_path, num_classes=3, length=args.num_seg)  
   
    model.load_state_dict(params['state_dict'])
    model.cuda()
    model.eval() 
    return model

def build_model_continue():
    modelLocation="./checkpoint/"+args.dataset+"_"+args.arch+"_split"+str(args.split)
    model_path = os.path.join(modelLocation,'model_best.pth.tar') 
    params = torch.load(model_path)
    print(modelLocation)
    if args.dataset=='ucf101':
        model=models.__dict__[args.arch](modelPath='', num_classes=101,length=args.num_seg)
    elif args.dataset=='hmdb51':
        model=models.__dict__[args.arch](modelPath='', num_classes=51,length=args.num_seg)
   
    model.load_state_dict(params['state_dict'])
    model = model.cuda()
    optimizer = AdamW(model.parameters(), lr= args.lr, weight_decay=args.weight_decay)
    optimizer.load_state_dict(params['optimizer'])
    for param_group in optimizer.param_groups:
        lr = param_group['lr'] 
    
    startEpoch = params['epoch']
    best_prec = params['best_prec1']
    return model, startEpoch, optimizer, best_prec

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(train_loader, model, criterion, criterion2, optimizer, epoch,setMseCoeff,modality):
    batch_time = AverageMeter()
    lossesClassification = AverageMeter()
    lossesMSE = AverageMeter()
    lossesBatchSimilarity=AverageMeter()
    lossesSequenceSimilarity=AverageMeter()
    lossesRanking=AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    criterion3 = torch.nn.CosineSimilarity(dim = 2)
    msecoeff=torch.tensor(setMseCoeff).cuda()

    # switch to train mode
    model.train()

    end = time.time()
    optimizer.zero_grad()
    loss_mini_batch_classification = 0.0
    loss_mini_batch_MSE = 0.0
    loss_mini_batch_ranking = 0.0
    acc_mini_batch = 0.0
    acc_mini_batch_top3 = 0.0
    totalSamplePerIter=0
    for i, (inputs, targets) in enumerate(train_loader):

        inputs = inputs.to(device)
        targets = targets.to(device)
        output, sequenceOut = model(inputs)

               
        prec1, prec3 = accuracy(output.data, targets, topk=(1, 3))
        acc_mini_batch += prec1.item()
        acc_mini_batch_top3 += prec3.item()
        
        
        #lossRanking = criterion(out_rank, targetRank)
        lossRanking=torch.tensor([0]).cuda()

        lossClassification = criterion(output, targets)
        lossMSE = criterion2(inputs, sequenceOut)
        #lossMSE = torch.mean(1 - criterion3(input_vectors,sequenceOut))
        
        lossRanking = lossRanking / args.iter_size
        lossClassification = lossClassification / args.iter_size
        lossMSE = lossMSE / args.iter_size
        
        #totalLoss=lossMSE
        
        totalLoss=lossClassification 
        #totalLoss = lossMSE * torch.tensor(20).cuda() + lossClassification 
        loss_mini_batch_classification += lossClassification.data.item()
        loss_mini_batch_MSE += lossMSE.data.item()
        loss_mini_batch_ranking += lossRanking.data.item()
        totalLoss.backward()
        totalSamplePerIter +=  output.size(0)
        if (i+1) % args.iter_size == 0:
            # compute gradient and do SGD step
            optimizer.step()
            optimizer.zero_grad()
            lossesClassification.update(loss_mini_batch_classification, totalSamplePerIter)
            lossesMSE.update(loss_mini_batch_MSE, totalSamplePerIter)
            lossesRanking.update(loss_mini_batch_ranking,totalSamplePerIter)
            top1.update(acc_mini_batch/args.iter_size, totalSamplePerIter)
            top3.update(acc_mini_batch_top3/args.iter_size, totalSamplePerIter)
            batch_time.update(time.time() - end)
            end = time.time()
            loss_mini_batch_classification = 0
            loss_mini_batch_MSE = 0
            loss_mini_batch_ranking = 0
            acc_mini_batch = 0
            acc_mini_batch_top3 = 0.0
            totalSamplePerIter = 0.0
    
            
        if (i+1) % args.print_freq == 0:
            print('[%d] time: %.3f loss: %.4f' %(i,batch_time.avg,lossesClassification.avg))
#        if (i+1) % args.print_freq == 0:
#
#            print('Epoch: [{0}][{1}/{2}]\t'
#                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                  'Classification Loss {lossClassification.val:.4f} ({lossClassification.avg:.4f})\t'
#                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
#                  'Prec@3 {top3.val:.3f} ({top3.avg:.3f})\n'
#                  'MSE Loss {lossMSE.val:.4f} ({lossMSE.avg:.4f})\t'
#                  'Batch Similarity Loss {lossBatchSimilarity.val:.4f} ({lossBatchSimilarity.avg:.4f})\t'
#                  'Sequence Similarity Loss {lossSequenceSimilarity.val:.4f} ({lossSequenceSimilarity.avg:.4f})'.format(
#                   epoch, i+1, len(train_loader)+1, batch_time=batch_time, lossClassification=lossesClassification,lossMSE=lossesMSE,
#                   lossBatchSimilarity = lossesBatchSimilarity , lossSequenceSimilarity=lossesSequenceSimilarity,
#                   top1=top1, top3=top3))
            
#    print(' * Epoch: {epoch} Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f} Classification Loss {lossClassification.avg:.4f} MSE Loss {lossMSE.avg:.4f} '
#          'Batch Similarity Loss {lossBatchSimilarity.avg:.4f} Sequence Similarity Loss {lossSequenceSimilarity.avg:.4f} Ranking Loss {lossRanking.avg:.4f}\n'
#          .format(epoch = epoch, top1=top1, top3=top3, lossClassification=lossesClassification,lossMSE=lossesMSE,
#                  lossBatchSimilarity = lossesBatchSimilarity , 
#                  lossSequenceSimilarity=lossesSequenceSimilarity), 
#                  lossRanking = lossesRanking) 
          
    print(' * Epoch: {epoch} Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f} Classification Loss {lossClassification.avg:.4f} MSE Loss {lossMSE.avg:.4f} '
          ' Ranking Loss {lossRanking.avg:.4f}\n'
          .format(epoch = epoch, top1=top1, top3=top3, lossClassification=lossesClassification,lossMSE=lossesMSE,
                  lossRanking = lossesRanking))
          
    writer.add_scalar('data/classification_loss_training', lossesClassification.avg, epoch)
    writer.add_scalar('data/mse_loss_training', lossesMSE.avg, epoch)
    writer.add_scalar('data/batchSimilarity_loss_training', lossesBatchSimilarity.avg, epoch)
    writer.add_scalar('data/sequenceSimilarity_loss_training', lossesSequenceSimilarity.avg, epoch)
    writer.add_scalar('data/total_loss_training', lossesMSE.avg+lossesClassification.avg+lossesBatchSimilarity.avg+lossesSequenceSimilarity.avg, epoch)
    writer.add_scalar('data/top1_training', top1.avg, epoch)
    writer.add_scalar('data/top3_training', top3.avg, epoch)
def validate(val_loader, model, criterion,criterion2,modality):
    batch_time = AverageMeter()
    lossesClassification = AverageMeter()
    lossesMSE = AverageMeter()
    lossesRanking=AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    criterion3 = torch.nn.CosineSimilarity(dim = 2)
    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
                

            inputs = inputs.to(device)
            targets = targets.to(device)
    
            # compute output
            if modality == 'both':
                output_rgb, output_flow, input_vectors, sequenceOut, maskSample = model(inputs)
            else:
                output, sequenceOut = model(inputs)
                
#            input_vectors_rank=input_vectors.view(-1,input_vectors.shape[-1])
#            targetRank=torch.tensor(range(args.num_seg)).repeat(input_vectors.shape[0]).cuda()
#            rankingFC = nn.Linear(input_vectors.shape[-1], args.num_seg).cuda()
#            out_rank = rankingFC(input_vectors_rank)
#            
#            lossRanking = criterion(out_rank, targetRank)
            
            lossRanking=torch.tensor([0]).cuda()

            lossClassification = criterion(output, targets)
            lossMSE = criterion2(inputs, sequenceOut)
            #lossMSE = torch.mean(1 - criterion3(input_vectors,sequenceOut))

            # measure accuracy and record loss
            prec1, prec3 = accuracy(output.data, targets, topk=(1, 3))
            
            lossesClassification.update(lossClassification.data.item(), output.size(0))
            lossesMSE.update(lossMSE.data.item(), output.size(0))
            lossesRanking.update(lossRanking.data.item(), output.size(0))
            
            top1.update(prec1.item(), output.size(0))
            top3.update(prec3.item(), output.size(0))
    
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    
#            if i % args.print_freq == 0:
#                print('Test: [{0}/{1}]\t'
#                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                      'Classification Loss {lossClassification.val:.4f} ({lossClassification.avg:.4f})\t'
#                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
#                      'Prec@3 {top3.val:.3f} ({top3.avg:.3f})\n'
#                      'MSE Loss {lossMSE.val:.4f} ({lossMSE.avg:.4f})\t'
#                      'Batch Similarity Loss {lossBatchSimilarity.val:.4f} ({lossBatchSimilarity.avg:.4f})\t'
#                      'Sequence Similarity Loss {lossSequenceSimilarity.val:.4f} ({lossSequenceSimilarity.avg:.4f})'.format(
#                       i, len(val_loader), batch_time=batch_time, lossClassification=lossesClassification,lossMSE=lossesMSE,
#                       lossBatchSimilarity = lossesBatchSimilarity , lossSequenceSimilarity=lossesSequenceSimilarity,
#                       top1=top1, top3=top3))
    
        print(' * * Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f} Classification Loss {lossClassification.avg:.4f} MSE Loss {lossMSE.avg:.4f} '
              ' Ranking Loss {lossRanking.avg:.4f}\n' 
              .format(top1=top1, top3=top3, lossClassification=lossesClassification,lossMSE=lossesMSE,
                      lossRanking = lossesRanking))

    return top1.avg, top3.avg, lossesClassification.avg, lossesMSE.avg

def save_checkpoint(state, is_best, filename, resume_path):
    cur_path = os.path.join(resume_path, filename)
    torch.save(state, cur_path)
    best_path = os.path.join(resume_path, 'model_best.pth.tar')
    if is_best:
        shutil.copyfile(cur_path, best_path)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 150 epochs"""

    decay = 0.1 ** (sum(epoch >= np.array(args.lr_steps)))
    lr = args.lr * decay
    print("Current learning rate is %4.6f:" % lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
        
def adjust_mse_coeff(mseCoeffStart, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 150 epochs"""
    setMseCoeff=mseCoeffStart*0.90**(epoch)
    
    print("Current mse coeff rate is %4.6f:" % setMseCoeff)
    return setMseCoeff


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
