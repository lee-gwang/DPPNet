from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torchvision import datasets

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils.utils as util
from utils.rcp_utils import DPPConv2d, DynConv2d, DynConv2d2
from utils.utils import GradualWarmupScheduler

import numpy as np
import random
import os, time, sys
import argparse
from model import resnet, dwpresnet, dynresnet, dynresnet_fine
from copy import deepcopy

import math
#amp
from torch.cuda.amp import autocast, GradScaler
import torch.multiprocessing
from torchsummary import summary
from thop import profile


torch.multiprocessing.set_sharing_strategy('file_system')

#torch.manual_seed(123123)
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True # True is deterministic
    torch.backends.cudnn.benchmark = True 

#----------------------------
# Argument parser.
#----------------------------
parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
# model
parser.add_argument('--dataset', '-ds', type=str, default='cifar10', help='dataset')
parser.add_argument('--arch_model', '-am', type=str, default='resnet18', help='architecture models')

parser.add_argument('--lr', '-lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('--min_lr', '-mlr', type=float, default=1e-4, help='min learning rate')

parser.add_argument('--scheduler', '-sch', type=str, default='multistep', help='lr_scheduler, [multistep, cosine]')
parser.add_argument('--warmup', '-warmup', action='store_true', help='scheduler warmup')
parser.add_argument('--pretrained', action='store_true', help='use pretrained models')


parser.add_argument('--wt_decay', '-wd', type=float, default=1e-4, help='weight decaying')
parser.add_argument('--epochs', '-e', type=int, default=90, help='epochs')
parser.add_argument('--batch_size', '-bs', type=int, default=128, help='batch size')
# KD
parser.add_argument('--lambda_KD',  type=float, default=0.5, help='KD ratio')
parser.add_argument('--alpha',  type=float, default=1e-4, help='threshold related')
parser.add_argument('--budget',  type=float, default=0.5, help='confidence of mask')

# amp
parser.add_argument('--amp', action='store_true', help='mixed precision')

parser.add_argument('--model_dir', '-md', type=str, default='default', help='saved model path')
parser.add_argument('--test', '-t', action='store_true', help='test only')
parser.add_argument('--resume', type=str, default=None, help='resume model.pth path')

parser.add_argument('--path', '-p', type=str, default=None, help='saved model path(pretrain_load)')

parser.add_argument('--which_gpus', '-gpu', type=str, default='0', help='which gpus to use')
parser.add_argument('--seed', type=int, default=42, help='seed')

args = parser.parse_args()

#########################
def print_args(args, logger=None):
    for k, v in vars(args).items():
        if logger is not None:
            logger.info('{:<16} : {}'.format(k, v))
        else:
            print('{:<16} : {}'.format(k, v))


#----------------------------
# Load the CIFAR-10 dataset.
#----------------------------

def load_dataset():
    if args.dataset=='imagenet':
        transform_train = transforms.Compose([
                            transforms.RandomSizedCrop(224),
                            transforms.RandomHorizontalFlip(),#ImageNetPolicy(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                        ])

        transform_test = transforms.Compose([
                            transforms.Resize(int(224/0.875)),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                        ])

        # dataset
        trainset = datasets.ImageNet('/home/data/imagenet/', split='train',download=False, transform=transform_train)
        valset = datasets.ImageNet('/home/data/imagenet/', split='train',download=False, transform=transform_test)
        testset = datasets.ImageNet('/home/data/imagenet/', split='val',download=False, transform=transform_test)

        # fixed index
        train_set_index = torch.randperm(len(trainset))
        if os.path.exists('/home/data/index_imagenet.pth'):
            print('!!!!!! Load train_set_index !!!!!!')
            train_set_index = torch.load('/home/data/index_imagenet.pth')

        num_sample_valid = 50000

        # loader
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, 
                                                sampler=torch.utils.data.sampler.SubsetRandomSampler(train_set_index[:-num_sample_valid]),
                                                num_workers=16, pin_memory=True)
        
        valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,  
                                            sampler=torch.utils.data.sampler.SubsetRandomSampler(train_set_index[-num_sample_valid:]),
                                            num_workers=16, pin_memory=True)

        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, num_workers=16 ,pin_memory=True)    

    elif args.dataset=='cifar10':
        # top
        transform_train = transforms.Compose([
                        transforms.Pad(4),
                        transforms.RandomCrop(32),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])

        transform_test = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])


        # dataset
        trainset = torchvision.datasets.CIFAR10(root='/home/data/cifar10_data', train=True,download=True, transform=transform_train)
        valset = torchvision.datasets.CIFAR10(root='/home/data/cifar10_data', train=True,download=True, transform=transform_test)
        testset = torchvision.datasets.CIFAR10(root='/home/data/cifar10_data', train=False, download=True, transform=transform_test)
        
        # fixed index
        train_set_index = torch.randperm(len(trainset))
        if os.path.exists('/home/data/index_c10.pth'):
            print('!!!!!! Load train_set_index !!!!!!')
            train_set_index = torch.load('/home/data/index_c10.pth')

        num_sample_valid = 5000

        # loader
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                    train_set_index[:-num_sample_valid]),
                                                num_workers=4, pin_memory=True)

        valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                    train_set_index[-num_sample_valid:]),
                                                num_workers=4, pin_memory=True)

        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                                num_workers=4, pin_memory=True)

    elif args.dataset == 'cifar100':
        transform_train = transforms.Compose([
                        transforms.Pad(4),
                        transforms.RandomCrop(32),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                        ])
        transform_test = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])

        # dataset
        trainset = torchvision.datasets.CIFAR100(root='/home/data/cifar100_data', train=True, download=True, transform=transform_train)
        valset = torchvision.datasets.CIFAR100(root='/home/data/cifar100_data', train=True, download=True, transform=transform_test)
        testset = torchvision.datasets.CIFAR100(root='/home/data/cifar100_data', train=False, download=True, transform=transform_test)
        
        # fixed index
        train_set_index = torch.randperm(len(trainset))
        if os.path.exists('/home/data/index_c100.pth'):
            print('!!!!!! Load train_set_index !!!!!!')
            train_set_index = torch.load('/home/data/index_c100.pth')

        num_sample_valid = 5000

        # loader
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                    train_set_index[:-num_sample_valid]),
                                                num_workers=4, pin_memory=True)

        valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                    train_set_index[-num_sample_valid:]),
                                                num_workers=4, pin_memory=True)

        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                                num_workers=4, pin_memory=True)
        



    return trainloader, valloader, testloader

#----------------------------
# Define the model.
#----------------------------
def generate_model(model_arch):
    # 0.1
    if model_arch == 'resnet20':
        return resnet.resnet20(dataset=args.dataset)
    elif model_arch == 'resnet32':
        return resnet.resnet32(dataset=args.dataset)
    elif model_arch == 'resnet56':
        return resnet.resnet56(dataset=args.dataset)

    elif model_arch == 'resnet18':
        return resnet.resnet18(dataset=args.dataset, pretrained=True)
    elif model_arch == 'resnet34':
        return resnet.resnet34(dataset=args.dataset, pretrained=True)  
    elif model_arch == 'resnet50':
        return resnet.resnet50(dataset=args.dataset, pretrained=True)


    # dwpresnet
    elif model_arch == 'dwpresnet20':
        return dwpresnet.dwpresnet20(dataset=args.dataset, pretrained=args.pretrained)
    elif model_arch == 'dwpresnet32':
        return dwpresnet.dwpresnet32(dataset=args.dataset, pretrained=args.pretrained)
    elif model_arch == 'dwpresnet56':
        return dwpresnet.dwpresnet56(dataset=args.dataset, pretrained=args.pretrained)

    elif model_arch == 'dwpresnet18':
        return dwpresnet.dwpresnet18(dataset=args.dataset, pretrained=args.pretrained)

    # dynresnet ( for pretrained)
    elif model_arch == 'dynresnet20':
        return dynresnet.dynresnet20(dataset=args.dataset)
    elif model_arch == 'dynresnet32':
        return dynresnet.dynresnet32(dataset=args.dataset)

    elif model_arch == 'dynresnet56':
        return dynresnet.dynresnet56(dataset=args.dataset)

    # Imagenet
    elif model_arch == 'dynresnet18':
        return dynresnet.dynresnet18(dataset=args.dataset)
    # fine-tuning
    elif model_arch == 'dynresnet20_fine':
        return dynresnet_fine.dynresnet20(dataset=args.dataset)
    elif model_arch == 'dynresnet32_fine':
        return dynresnet_fine.dynresnet32(dataset=args.dataset)
    elif model_arch == 'dynresnet18_fine':
        return dynresnet_fine.dynresnet18(dataset=args.dataset)
    elif model_arch == 'dynresnet34_fine':
        return dynresnet_fine.dynresnet34(dataset=args.dataset)
    elif model_arch == 'dynresnet56_fine':
        return dynresnet_fine.dynresnet56(dataset=args.dataset)
    else:
        raise NotImplementedError("Model architecture is not supported.")

#----------------------------
# Train the network.
#----------------------------


def train_model(net, device, logger):
    # define the loss function
    #seed_everything(42)

    # dataset
    print("Loading the data.")
    trainloader, valloader, testloader = load_dataset()

    #
    criterion = (nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss())
    
    

    if args.dataset=='imagenet':
        lr_decay_milestones = [args.epochs//3, args.epochs//3*2]
    else:
        lr_decay_milestones = [args.epochs//4*2, args.epochs//4*3] # epochs 160 ==>[80, 120]
    
    # initialize the optimizer
    optimizer = optim.SGD(net.parameters(), 
                          lr=0.005 if args.warmup else args.lr, 
                          momentum=0.9, weight_decay = args.wt_decay)

    if args.scheduler == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(
                        optimizer, 
                        milestones=lr_decay_milestones, 
                        gamma=0.1,
                        last_epoch=-1)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, 
                        T_max=args.epochs, 
                        eta_min = args.min_lr,
                        last_epoch=-1)

    if args.warmup:
        scheduler2 = GradualWarmupScheduler(optimizer, multiplier=20, total_epoch=5 if args.dataset=='imagenet' else 10, after_scheduler=scheduler)

    if args.resume is not None:
        checkpoint = torch.load(args.resume +'/last_model.pth')
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        resume_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']


    #amp
    scaler = GradScaler()
    if args.amp:
        print('mixed precision training!')
    else:
        print('normal training')
    best_acc, best_sparsity = 0 ,0
    
    ###########################
    # Training
    ###########################
    for epoch in range(args.epochs): # loop over the dataset multiple times
        if args.resume is not None:
            if resume_epoch+1 > epoch:
                continue
        # print(f'current epoch is {epoch}')

        # start training!
        net.train()
        #print('current learning rate = {}'.format(optimizer.param_groups[0]['lr']))
        logger.info(f"current learning rate = {optimizer.param_groups[0]['lr'] :.5f}")
        
        # each epoch
        start = time.time()
        for i, data in enumerate(tqdm(trainloader)):
            if args.amp:
                with autocast():
                    inputs, labels = data[0].to(device), data[1].to(device) # top data

                    optimizer.zero_grad()

                    outputs = net(inputs)
                    
                    # loss
                    loss = torch.FloatTensor([0.]).to(device)
                    loss += criterion(outputs, labels)


                    # amp
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                
                inputs, labels = data[0].to(device), data[1].to(device) # top data
                
                optimizer.zero_grad()
                outputs = net(inputs)
                
                # loss
                loss = torch.FloatTensor([0.]).to(device)
                loss += criterion(outputs, labels)

                loss.backward()
                optimizer.step()


        # update the learning rate
        scheduler2.step() if args.warmup else scheduler.step()

        logger.info('epoch {}'.format(epoch+1))
        acc, _=  test_accu(valloader, net, device, logger, macs, types='Val') # original size
        test_accu(testloader, net, device, logger, macs, types='Test')

        # last epoch save
        torch.save({'state_dict': net.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'scheduler' : scheduler.state_dict(),
                    'epoch' : epoch,
                    'best_acc' : best_acc}, 
                    
                    f'{args.model_folder}/last_model.pth')   

        if acc>best_acc:
            best_epoch = epoch+1
            best_acc = acc
            logger.info("Saving the best trained model.")
            torch.save({'state_dict':net.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scheduler' : scheduler.state_dict(),
                        'epoch' : best_epoch,
                        'best_acc' : best_acc}, 
            
            f'{args.model_folder}/best_model.pth')
    

    logger.info('Finished Training')
    logger.info(f'validation best {best_epoch} epoch, accuracy: {best_acc:.2f}')

    logger.info('############# Test set Accuracy #############')
    checkpoint = torch.load(f'{args.model_folder}/best_model.pth')
    net.load_state_dict(checkpoint['state_dict'])
    test_accu(testloader, net, device, logger, macs, types='Test')


#----------------------------
# Test accuracy.
#----------------------------

def test_accu(testloader, net, device, logger, macs, types='Test'):
    correct = 0
    predicted = 0
    total = 0.0

    # sparsity ##############################
    cnt_out = []
    cnt_full = []
    num_out = []
    num_full = []
    def _report_sparsity(m):
        classname = m.__class__.__name__
        if isinstance(m, DPPConv2d):
            num_out.append(m.num_out)
            num_full.append(m.num_full)
            m.num_out = torch.tensor([0])
            m.num_full = torch.tensor([0])
    ####################
    net.eval()

    with torch.no_grad():
        for data in tqdm(testloader):
            images, labels = data[0].to(device), data[1].to(device)

            outputs = net(images)

            _, predicted = torch.max(outputs.data, 1)

            correct += float(predicted.eq(labels.data).cpu().sum())
            total += float(labels.size(0))


            # sparsity
            net.apply(_report_sparsity)
            cnt_out.append(np.array(num_out))
            cnt_full.append(np.array(num_full))
            num_out = []
            num_full = []

        sparsity_list = [np.sum(cnt_full), np.sum(cnt_out)]
        
            
    logger.info(f'{types} ACC : {100*correct/total:.2f}%')
    
    return 100*correct/total, sparsity_list

import torch.nn.utils.prune as prune
def pruned_layers(net):
    tmps = []
    for n, conv in enumerate(net.modules()):
        if isinstance(conv, DynConv2d2):
            tmp_pruned = conv.weight.data.clone()
            original_size = tmp_pruned.size() # (out, ch, h, w) # 
            num_pattern, kernel_size = tmp_pruned.shape[0], tmp_pruned.shape[3]
            mask = ((F.sigmoid((tmp_pruned.unsqueeze(0).abs() - conv.threshold).mean(dim=(2,3))) - 0.5)>0.).float().view(num_pattern, 1, 1, kernel_size, kernel_size)
            mask = mask.expand(original_size)
            prune.custom_from_mask(conv, name='weight', mask=mask)
    return net

#----------------------------
# Main function.
#----------------------------

def main():
    # log
    model_folder = f'./saved_models/' + args.model_dir
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    else:
        assert False, 'exist folder name'
    setattr(args, 'model_folder', model_folder)
    logger = util.create_logger(model_folder, 'train', 'info')
    print_args(args, logger)
    ###
    os.environ["CUDA_VISIBLE_DEVICES"] = args.which_gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Available GPUs: {}".format(torch.cuda.device_count()))

    print("Create {} model.".format(args.arch_model))
    net = generate_model(args.arch_model)
    if len(args.which_gpus)>1:
        net = nn.DataParallel(net)

    temp = torch.load(args.path)['state_dict']
    net.load_state_dict(temp)
    net = pruned_layers(net)

    net.to(device)  
    
    # train
    print("Start training.")
    train_model(net, device, logger)
    

if __name__ == "__main__":
    seed_everything(args.seed)
    main()
