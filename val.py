import datetime, argparse, pprint
from fastai.vision import *
import warnings
import sklearn
import json
import numpy as np
import pandas as pd
import torch.nn.functional as F
import evaluation
import torch.backends.cudnn as cudnn
warnings.filterwarnings("ignore")

from _collections import OrderedDict
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms
from sklearn.metrics import classification_report
from isic import ISICDataset
from GCPLoss import GCPLoss
from isic_datafileload import load_datafile
from models import SimpleCNN
from tqdm import tqdm
from sklearn.metrics import classification_report

## Transforms
feat_dim = 512 # Hardcoded for resnet34
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

def validate(args,valid_dl, valid_ood,model, criterion):
    '''Returns testing accuracy on the In-Distribution test set and OOD performance
    on the OOD test set. Also returns the probability distributions of ID and OOD test sets.'''
    actuals = []
    running_corrects = 0
    _pred_k, _pred_u = [], []
    size=0
    count=0
    model.eval()

    
    with torch.no_grad():
        print ('Starting the In-Distribution Testing')
        for inputs, labels, _ in tqdm(valid_dl):
            count += 1
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            if args.loss == 'Softmax':
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                _pred_k.append(F.softmax(outputs,dim=1).data.cpu().numpy())
            elif args.loss == 'GCPLoss':
                x, y = model(images)
                logits, _ = criterion(x, y, targets)
                preds = logits.data.max(1)[1]
                _pred_k.append(logits.data.cpu().numpy())

            actuals.extend([i.item() for i in labels])
            running_corrects += torch.sum(preds == labels.data)

            size += inputs.size(0)
            
            # debug
            if args.debug_count > 0 and count >= args.debug_count:
                break
        
        print ('Starting the OOD Testing')
        count_out = 0
        for inputs, targets, _ in tqdm(valid_ood):
            count_out += 1
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            
            # Pass the inputs through the CNN model.
            if args.loss == 'Softmax':
                outputs = model(inputs)
                _pred_u.append(F.softmax(outputs,dim=1).data.cpu().numpy())
            elif args.loss == 'GCPLoss':
                x, y = model(images)
                logits, _ = criterion(x, y)
                _pred_u.append(logits.data.cpu().numpy())
            
            if args.debug == 1 and count_out >= 1:
                break

    _pred_k = np.concatenate(_pred_k, 0)
    _pred_u = np.concatenate(_pred_u, 0)

    # Out-of-Distribution detction evaluation
    x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
    results = evaluation.metric_ood(x1, x2)['Bas']
        
    return running_corrects, size, actuals, preds, results, _pred_k, _pred_u
        
    
def main(args,options):
    print (args)
    print('Resume from checkpoint {}...'.format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

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
    
    df = pd.read_csv(args.csv_file)

    # load data
    print('Load data from data dir: {} and csv file: {}'.format(args.data_dir, args.csv_file))
    df_in, df_out = load_datafile(args.data_dir,args.csv_file)

    l2i = {'MEL':0, 'NV':1, 'BCC':2, 'AK':3, 'BKL':4, 'SCC':5, 'DF': 6, 'VASC': 7}
    val_index = np.load('isic_val_index_org.npy')
    print ("Total test samples",len(val_index))
    df_test = df_in.iloc[val_index]

    test_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Dataloader
    valid_ds = ISICDataset(args.data_dir, df_test, test_transform, size=args.image_size, is_train=False, test_mode=True)
    valid_ood_data = ISICDataset(args.data_dir, df_out, test_transform, size=args.image_size, is_train=False, test_mode=True)
    valid_dl = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    valid_ood_dl = DataLoader(valid_ood_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print('Valid size: {} OOD size: {}'.format(len(valid_ds),len(valid_ood_data)))

    # Define the criterion function.
    if args.loss == 'Softmax':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'GCPLoss':
        criterion = GCPLoss(**options)
        criterion = criterion.cuda()
    
    # load model
    print('Build and load model parameters...')
    args.num_classes = 6
    print('total classes is {}'.format(args.num_classes))
    model = SimpleCNN(args.network,args.num_classes)
    model = nn.DataParallel(model).to(args.device)
    model.to(args.device)
    model.load_state_dict(checkpoint['model'])
    if args.loss == 'GCPLoss':
        criterion.load_state_dict(torch.load(args.checkpoint)['criterion'])
    print ("Loaded model")

    # run validation
    print('Run validation...')

    running_corrects, size, actuals, preds, ood_results, probs_k, probs_u = validate(args,valid_dl, valid_ood_dl, model, criterion)

    #Save some files
    final_preds = preds.cpu().numpy()
    result = {}
    test_acc = running_corrects.double() / size
    print ("ID - Testing acc",test_acc)
    print ("OOD - AUROC",ood_results['AUROC'])                 
    
    os.makedirs(os.path.join(args.output_dir,args.output_filename.split('.')[0]), exist_ok=True)

    stats = classification_report(actuals, final_preds, output_dict=True)
    print (stats)
    torch.save(stats, os.path.join(args.output_dir, args.output_filename.split('.')[0],'_reports.pt'))
    df_stats = pd.DataFrame(stats).transpose()
    df_stats.to_csv(os.path.join(args.output_dir,args.output_filename.split('.')[0],'id_reports.csv'))

    with open(os.path.join(args.output_dir, args.output_filename.split('.')[0],'in_probs_dist.npy'), 'wb') as f:
        np.save(f, probs_k)
    f.close()

    with open(os.path.join(args.output_dir, args.output_filename.split('.')[0],'out_probs_dist.npy'), 'wb') as f:
        np.save(f, probs_u)
    
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate model with the test dataset')
    # data
    parser.add_argument('--data-dir', default='../../../data/isic2019/ISIC_2019_Training_Input', help='data directory')
    parser.add_argument('--csv-file', default='ISIC_2019_Training_GroundTruth.csv', help='path to csv file')
    parser.add_argument('--checkpoint', default='runs/exp/checkpoints/checkpoint.pth', help='path to checkpoint file')
    parser.add_argument('--device', default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                        help='device {cuda:0, cpu}')
    parser.add_argument('--output-dir', default='results', help='where to save results')
    parser.add_argument('--output-filename', type=str, default=None, help='output file name, default: val_YYYYMMDD.csv')
    # data params
    parser.add_argument('--image-size', type=int, default=320, help='image size to the model')
    parser.add_argument('--num-workers', type=int, default=8, help='number of data loader workers')
    parser.add_argument('--batch-size', type=int, default=128, help='mini batch size')
    # other
    parser.add_argument('--debug', default=False, action='store_true', help='turn on debug mode')
    parser.add_argument('--debug-count', type=int, default=0, help='# of minibatchs for fast testing, 0 to disable')
    parser.add_argument('--network', type=str, default='res34', help='res34 or res50')
    parser.add_argument('--loss',type=str, default='Softmax', help='GCPLoss or Softmax')
    parser.add_argument('--temp', type=float, default=1.0, help="temp")
    parser.add_argument('--weight-pl', type=float, default=0.1, help="weight for RPL loss")

    options = vars(parser.parse_args())
    main(parser.parse_args(),options)