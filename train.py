import torch
import pandas as pd
import numpy as np
import os
import sys
import argparse
import fastai
import torch.nn as nn
import time, datetime
import torchvision
import torch.nn.functional as F
import evaluation as evaluation
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader
from isic import ISICDataset
from schedulers import get_scheduler
from fastai import *
from fastai.vision import *
from fastai.vision import get_transforms
from models import SimpleCNN
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from GCPLoss import GCPLoss
from isic_datafileload import load_datafile
from torch.autograd import Variable
import warnings
warnings.filterwarnings("ignore")

#Hardcoded parameters
feat_dim = 512 #Resnet34 features
## Transforms
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    '''Returns mixup loss for Cross Entropy criterion only between two samples'''
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_one_epoch(args, writer, model, criterion, data_loader, optimizer, epoch, max_norm=0.0):
    '''Returns train loss after training the model for one epoch
    All different kind of input arguments regarding the loss function
    and whether to do mixup or not are covered in this section'''
    model.train()
    criterion.train()
    count = 0
    for images, targets in tqdm(data_loader):
        count += 1
        images = images.to(args.device)
        targets = targets.to(args.device)
        
        # Pass the inputs through the CNN model.
        if args.loss == 'Softmax':
            outputs = model(images)
            if args.mixup == 1:
                inputs, targets_a, targets_b, lam = mixup_data(images, targets,
                                                            args.alpha)
                inputs, targets_a, targets_b = map(Variable, (inputs,
                                                            targets_a, targets_b))
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                loss = criterion(outputs, targets)
        elif args.loss == 'GCPLoss':
            if args.mixup == 1:
                inputs, targets_a, targets_b, lam = mixup_data(images, targets,
                                                            args.alpha)
                inputs, targets_a, targets_b = map(Variable, (inputs,
                                                            targets_a, targets_b))
                x, y = model(inputs)
                logits, loss = criterion(x, y, labels=targets, targets_a=targets_a, targets_b=targets_b, lam=lam, mixup=args.mixup)
            else:
                x, y = model(images)
                logits, loss = criterion(x, y, targets, args.mixup)
        
        if writer is not None and count % args.log_steps == 1:
            writer.add_scalars('Loss/train', {'loss': loss}, epoch*len(data_loader)+count)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.debug == 1 and count >= 1:
            return loss

    return loss  

@torch.no_grad()
def evaluate(args, writer, model, criterion, data_loader, data_loader_ood, epoch):
    '''Returns validation loss after validating the model on the valid set;
    Also returns the OOD detection results after validating on the OOD set.'''
    model.eval()
    criterion.eval()
    count = 0
    size = 0
    running_loss = 0.0
    running_corrects = 0
    _pred_k, _pred_u, _labels = [], [], []

    for images, targets in tqdm(data_loader):
        count += 1
        images = images.to(args.device)
        targets = targets.to(args.device)
        
        # Pass the inputs through the CNN model.
        if args.loss == 'Softmax':
            outputs = model(images)
            loss = criterion(outputs, targets)
            _, preds = torch.max(outputs, 1)
            _pred_k.append(F.softmax(outputs,dim=1).data.cpu().numpy())
        elif args.loss == 'GCPLoss':
            x, y = model(images)
            logits, loss = criterion(x, y, targets)
            preds = logits.data.max(1)[1]
            _pred_k.append(logits.data.cpu().numpy())
        
        # Calculate the batch loss.
        running_loss += loss.item() * images.size(0)
        
        running_corrects += torch.sum(preds == targets.data)
        
        size += images.size(0)

        if args.debug == 1 and count >=1:
            break

    count_out = 0
    for images, targets, _ in tqdm(data_loader_ood):
        count_out += 1
        images = images.to(args.device)
        targets = targets.to(args.device)
        
        # Pass the inputs through the CNN model.
        if args.loss == 'Softmax':
            outputs = model(images)
            _pred_u.append(F.softmax(outputs,dim=1).data.cpu().numpy())
        elif args.loss == 'GCPLoss':
            x, y = model(images)
            logits, _ = criterion(x, y)
            _pred_u.append(logits.data.cpu().numpy())
        
        if args.debug == 1 and count_out >= 1:
            break
        
    # end epoch
    epoch_loss = running_loss / size
    epoch_acc = running_corrects.double() / size
    
    disp_str = 'Epoch {} Losses: {:.4f}'.format(epoch+1, epoch_loss)

    _pred_k = np.concatenate(_pred_k, 0)
    _pred_u = np.concatenate(_pred_u, 0)

    # Out-of-Distribution detction evaluation
    x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
    results = evaluation.metric_ood(x1, x2)['Bas']
    
    if writer is not None:
        writer.add_scalars('Loss/valid', {'loss': epoch_loss}, epoch)
        writer.add_scalars('Accuracy/valid', {'acc': epoch_acc}, epoch)
        writer.add_text('Log/valid', disp_str, epoch)
        
    return epoch_loss, epoch_acc, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training on ISIC dataset')
    # data
    parser.add_argument('--img-size', type=int, default=224, help='Training image size to be passed to the network')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--num-workers', type=int, default=8, help='number of loader workers')
    parser.add_argument('--device', default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                        help='device {cuda:0, cpu}')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--log-dir', default='runs_new/runs/', help='where to store results')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
    parser.add_argument('--log_steps', type=int, default=100, help='Logging at steps')
    parser.add_argument('--network', type=str, default='res34', help='res34 or res50')
    parser.add_argument('--debug',type=int, default=0, help='Debug mode: 1')
    parser.add_argument('--dropout',type=float,default=0.,help='Dropout rate for Transformer')
    parser.add_argument('--scheduler',type=str, default='cosine_warm_restarts_warmup',help='Type of scheduler')
    parser.add_argument('--num-restarts',type=int,default=2, help='Number of restarts for scheduler')
    parser.add_argument('--checkpoint_path',type=str, default=None, help='Checkpoint path for resuming the training')
    parser.add_argument('--loss',type=str, default='Softmax', help='GCPLoss or Softmax')
    parser.add_argument('--temp', type=float, default=1.0, help="temp")
    parser.add_argument('--weight-pl', type=float, default=0.1, help="weight for RPL loss")
    parser.add_argument('--mixup',type=int,default=0, help = 'Option 0: No mixup, Option 1: With mixup')
    parser.add_argument('--alpha', default=1., type=float, help='mixup interpolation coefficient (default: 1)')
    args = parser.parse_args()

    print (args)
    options = vars(args)
    use_gpu = torch.cuda.is_available()
    options.update(
        {
            'feat_dim': feat_dim,
            'use_gpu': use_gpu
        }
    )
    if use_gpu:
        cudnn.benchmark = True
    else:
        print("Currently using CPU")

    #Image folder
    image_folder = '../../../data/isic2019/ISIC_2019_Training_Input'

    # CSV file path
    datafile_path = 'ISIC_2019_Training_GroundTruth.csv'

    #load data
    df_in, df_out = load_datafile(image_folder,datafile_path)
    
    #Data pre-preocessing
    train_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomCrop(args.img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    test_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    # Dataloader
    train_data = ISICDataset(image_folder, df_in, train_transform, size=args.img_size, is_train=True, test_mode=False)
    valid_data = ISICDataset(image_folder, df_in, test_transform, size=args.img_size, is_train=False, test_mode=False)
    valid_ood_data = ISICDataset(image_folder, df_out, test_transform, size=args.img_size, is_train=False, test_mode=True)
    train_dl = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    valid_dl = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_ood_dl = DataLoader(valid_ood_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)        

    print('Train size: {}, valid size: {}'.format(len(train_data), len(valid_data)))

    # Initialize the model
    # model
    print('Build model')
    print("Using", torch.cuda.device_count(), "GPUs.")
    args.num_classes = 6
    options.update(
        {
            'num_classes': args.num_classes
        }
    )
    print('total classes is {}'.format(args.num_classes))
    model = SimpleCNN(args.network,args.num_classes,args.loss)
    model = nn.DataParallel(model).to(args.device)
    print ("Loaded model")
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # Define the loss function.
    if args.loss == 'Softmax':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'GCPLoss':
        criterion = GCPLoss(**options)
        criterion = criterion.cuda()
    
    if args.loss == 'Softmax':
        param_dicts = [{"params": [p for n, p in model.named_parameters() if p.requires_grad]}]

    elif args.loss == 'GCPLoss':
        param_dicts = [{"params": [p for n, p in model.named_parameters() if p.requires_grad]},{'params': criterion.parameters()}]

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = get_scheduler(optimizer,args)

    args.log_dir = 'runs_openset_isic' + '/exp_' + args.network + '_' + str(args.loss) + '_mixup' + str(args.mixup)
    
    #Checkpoint saving for models best at Openset and best at ID classification
    writer = SummaryWriter(log_dir = args.log_dir)
    output_dir = Path(writer.log_dir)
    checkpoint_path_auroc = Path(os.path.join(output_dir,'auroc', 'checkpoints'))
    os.makedirs(checkpoint_path_auroc, exist_ok=True)
    checkpoint_path_auroc = checkpoint_path_auroc / 'checkpoint.pth'
    checkpoint_path_val = Path(os.path.join(output_dir,'val', 'checkpoints'))
    os.makedirs(checkpoint_path_val, exist_ok=True)
    checkpoint_path_val = checkpoint_path_val / 'checkpoint.pth'
    args.start_epoch = 0

    best_valid_loss, best_auroc = 100, 0.0
    best_monitor_loss = None
    for epoch in range(args.start_epoch, args.epochs):
        print('\nEpoch', epoch)
        epoch_start_time = time.time()

        #Train one epoch
        print ('\nTrain: ')
        train_loss = train_one_epoch(args, writer, model, criterion, train_dl, optimizer, epoch)
       
        # evaluate
        print('\nEvaluate: ')
        valid_loss, valid_acc, results = evaluate(args, writer, model, criterion, valid_dl, val_ood_dl, epoch)
        checkpoint_paths_val, checkpoint_paths_auroc = [], []

        print ("\n Train loss:",train_loss.item(),"Valid loss:",valid_loss, "Valid acc: ", valid_acc, "AUROC: ",results['AUROC'])

        lr_scheduler.step(epoch=epoch)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            checkpoint_paths_val.append(checkpoint_path_val)

        if results['AUROC'] > best_auroc:
            best_auroc = results['AUROC']
            checkpoint_paths_auroc.append(checkpoint_path_auroc)
        
        #Save checkpoints every 10 epochs
        if (epoch + 1) % 10 == 0: 
            checkpoint_paths_auroc.append(output_dir / f'checkpoint{epoch:03}.pth')
            checkpoint_paths_val.append(output_dir / f'checkpoint{epoch:03}.pth')

        if args.loss == 'Softmax':
            for cp in checkpoint_paths_auroc:
                print('Save checkpoint {}'.format(cp))
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch
                }, cp)

            for cp in checkpoint_paths_val:
                print('Save checkpoint {}'.format(cp))
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch
                }, cp)

        elif args.loss == 'GCPLoss':
            for cp in checkpoint_paths_auroc:
                print('Save checkpoint {}'.format(cp))
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'criterion': criterion.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch
                }, cp)

            for cp in checkpoint_paths_val:
                print('Save checkpoint {}'.format(cp))
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'criterion': criterion.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch
                }, cp)

        epoch_total_time = time.time() - epoch_start_time
        epoch_total_time_str = str(datetime.timedelta(seconds=int(epoch_total_time)))
        print('Epoch training time {}\n'.format(epoch_total_time_str))

    if writer is not None: writer.close()