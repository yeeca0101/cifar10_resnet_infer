import sys
sys.path.append('../')
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import resnet
from lightvit import lightvit_tiny_cifar
# from resnet import sASN
from experiments.activation.acts import *
from experiments.data.datasets import CIFAR10,CIFAR100
from utils import MetricsTracker,plot_and_save_results

model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))
model_names.append('lightvit')
print(model_names)

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10/100 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32)')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save_folder', dest='save_folder',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
parser.add_argument('--act', dest='activation function',
                    help='lower string. ex. sASN : sasn',
                    default='none', type=str)
parser.add_argument('-n', dest='n_classes',
                    help='n',
                    default=10, type=int)
parser.add_argument('--repeat', default=1, type=int, help='number of repetitive training')

args = parser.parse_args()

def main(act,act_name,i):
    
    dataset_class = CIFAR10 if args.n_classes == 10 else CIFAR100
    save_folder = args.save_folder
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
        print('make dirs ',save_folder)
    log_dir_name = args.save_folder.split('/')[1]
    log_dir=os.path.join('logs', f'{log_dir_name}',f'{args.arch}',f'{act_name}_{i}')
    print('log_dir : ',log_dir)
    writer = SummaryWriter(logdir=log_dir)

    # TODO:parse activation
    args.act = act
    print('args:',args)
    # Check the save_dir exists or not

    # model state #########################################################
    if args.arch == 'lightvit':
        model= lightvit_tiny_cifar(act_layer=args.act,num_classes=args.n_classes)
        model = torch.nn.DataParallel(model)
    else:
        model = torch.nn.DataParallel(resnet.__dict__[args.arch](act=args.act,num_classes=args.n_classes))
    model_type = args.arch
    model.cuda()
    cudnn.benchmark = True

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
            dataset_class(train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]),),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
            dataset_class(train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer ############################
    criterion = nn.CrossEntropyLoss().cuda()

    if args.half:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[100, 150], last_epoch=args.start_epoch - 1)

    if args.arch in ['resnet1202', 'resnet110','lightvit']:
        # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this setup it will correspond for first epoch.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr*0.1

    if args.evaluate:
        validate(val_loader, model, criterion)
        return
    
    print('act:',args.act)
    metrics_tracker = MetricsTracker()
    best_acc,best_state = -1.,{}
    best_model_file_name = ""

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train_re = train(train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        val_re = validate(val_loader, model, criterion)

        metrics_tracker.update(train_loss=train_re['loss'], train_acc=train_re['top1'],val_loss=val_re['loss'], val_acc =val_re['top1'])

        # remember best prec@1 and save checkpoint
        acc = val_re['top1']
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': model.state_dict(),
                'acc': acc, # will be removed
                'epoch': epoch,
                'act' : act,
                'top1':val_re['top1'],
                'top5':val_re['top5']
            }
            if best_model_file_name:
                try:
                    # Attempt to remove the previous best model's checkpoint
                    os.remove(f'./{save_folder}/{best_model_file_name}')
                except:
                    pass
            best_model_file_name = f'{i}_ckpt_{model_type}_{dataset_class.__name__}_{act_name}_{val_re["top1"]:.2f}.pth'
            torch.save(state, f'./{save_folder}/{best_model_file_name}')
            best_acc = acc
            best_state = state
        
        metrics_tracker.plot_and_save('./plot_re/',f'curr_{args.act}_{args.arch}.png')
        # Logging to TensorBoard
        writer.add_scalar('Loss/train', train_re['loss'], epoch)
        writer.add_scalar('Accuracy-top1/train', train_re['top1'], epoch)
        writer.add_scalar('Loss/val', val_re['loss'], epoch)
        writer.add_scalar('Accuracy-top1/val', val_re['top1'], epoch)
        if args.n_classes == 100:
            writer.add_scalar('Accuracy-top5/val', val_re['top5'], epoch)

    print(best_state['act'],' ',best_state['epoch'],' ',best_state['acc'])
    
    res = {
        'act':name,
        'model':args.arch,
        'train_loss':metrics_tracker.train_loss,
        'train_acc':metrics_tracker.train_acc,
        'val_loss':metrics_tracker.val_loss,
        'val_acc':metrics_tracker.val_acc,
    }
    return res


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        targets = target.cuda()
        inputs = input.cuda()
        if args.half:
            inputs = inputs.half()
            targets = target.half()

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        if args.n_classes == 10:
            prec1 = accuracy(outputs,targets,topk=(1,))[0]
        else:
            prec1,prec5 = accuracy(outputs,targets,topk=(1,5))
            top5.update(prec5.item(),inputs.size(0))
        top1.update(prec1.item(),inputs.size(0))
        losses.update(loss.item(),inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
            
    results = {
        'loss':losses.avg,
        'top1':top1.avg,
        'top5':top5.avg if args.n_classes == 100 else 0.,
    }

    return results

def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            inputs = input.cuda()
            targets = target.cuda()

            if args.half:
                inputs = inputs.half()

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

             # top-1 top-5 acc
            if args.n_classes == 10:
                prec1 = accuracy(outputs.float().data,targets,topk=(1,))[0]
            else:
                prec1,prec5 = accuracy(outputs.float().data,targets,topk=(1,5))
                top5.update(prec5.item(),inputs.size(0))
            top1.update(prec1.item(),inputs.size(0))
            losses.update(loss.item(),inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))
    
    results = {
        'top1':top1.avg,
        'top5':top5.avg if args.n_classes==100 else 0.,
        'loss':losses.avg,
    }

    return results

# def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
#     """
#     Save the training model
#     """
#     torch.save(state, filename)

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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    acts=get_activations()
    # acts['PReLU']=nn.PReLU()
    print(acts.keys())
    comp_dict = {}
    for i in range(1,args.repeat+1):
        for name,act in acts.items():
            print(f'act : {name}')
            res = main(act,name,i)
            comp_dict[name]=res
            comp_dict['model']=res['model']
        plot_and_save_results(comp_dict,os.path.join('results',f'{args.save_folder}',comp_dict['model']))