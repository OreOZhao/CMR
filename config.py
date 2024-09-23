import os
import random
import torch
import argparse
import warnings

import torch.backends.cudnn as cudnn

import datetime

tasks = ['FB15k237_ind', 'WN18RR_ind', 'WN9_ind']
parser = argparse.ArgumentParser(description='SimKGC arguments')
parser.add_argument('--pretrained-model', default='./PLMs/bert-base-uncased', type=str, metavar='N',
                    help='path to pretrained model')
parser.add_argument('--task', default='FB15k237_ind', choices=tasks, type=str, metavar='N',
                    help='dataset name')
parser.add_argument('--train-path', default='', type=str, metavar='N',
                    help='path to training data')
parser.add_argument('--valid-path', default='', type=str, metavar='N',
                    help='path to valid data')
parser.add_argument('--model-dir', default='', type=str, metavar='N',
                    help='path to model dir')
parser.add_argument('--warmup', default=400, type=int, metavar='N',
                    help='warmup steps')
parser.add_argument('--max-to-keep', default=5, type=int, metavar='N',
                    help='max number of checkpoints to keep')
parser.add_argument('--grad-clip', default=10.0, type=float, metavar='N',
                    help='gradient clipping')
parser.add_argument('--pooling', default='mean', type=str, metavar='N',
                    help='bert pooling')
parser.add_argument('--dropout', default=0.1, type=float, metavar='N',
                    help='dropout on final linear layer')
parser.add_argument('--use-amp', action='store_true',
                    help='Use amp if available')
parser.add_argument('--t', default=0.05, type=float,
                    help='temperature parameter')
parser.add_argument('--use-link-graph', action='store_true',
                    help='use neighbors from link graph as context')
parser.add_argument('--eval-every-n-step', default=10000, type=int,
                    help='evaluate every n steps')
parser.add_argument('--pre-batch', default=2, type=int,
                    help='number of pre-batch used for negatives')
parser.add_argument('--pre-batch-weight', default=0.5, type=float,
                    help='the weight for logits from pre-batch negatives')
parser.add_argument('--additive-margin', default=0.0, type=float, metavar='N',
                    help='additive margin for InfoNCE loss function')
parser.add_argument('--finetune-t', action='store_true',
                    help='make temperature as a trainable parameter or not')
parser.add_argument('--max-num-tokens', default=50, type=int,
                    help='maximum number of tokens')
parser.add_argument('--use-self-negative', action='store_true',
                    help='use head entity as negative')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=768, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr-scheduler', default='linear', type=str,
                    help='Lr scheduler to use')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--seed', default=202303, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--mm', action='store_true',
                    help='CustomMM Model')
parser.add_argument('--prefix', default=0, type=int,
                    help='CustomMM Model with visual prefix fusion length')
# only used for evaluation
parser.add_argument('--is-test', action='store_true',
                    help='is in test mode or not')
parser.add_argument('--eval-model-path', default='', type=str, metavar='N',
                    help='path to model, only used for evaluation')
parser.add_argument('--early-stopping', default=3, type=int, metavar='N',
                    help='number of epochs to early stop when no increase in valid acc@1')
parser.add_argument('--pretrained-model-path', default='', type=str, metavar='N',
                    help='path to pre-trained model')
parser.add_argument('--knn_topk', default=0, type=int,
                    help='topk in kNN search.')
parser.add_argument('--knn_lambda', default=0.2, type=float,
                    help='p_knn ratio in kNN.')

args = parser.parse_args()

if args.mm:
    args.pretrained_vit = './PLMs/vit-base-patch16-224'


# assert not args.train_path or os.path.exists(args.train_path)
assert args.pooling in ['cls', 'mean', 'max']
assert args.task.lower() in ['wn18rr', 'wn18rr_ind', 'wn9_ind', 'wn18', 'wn9', 'fb15k237', 'fb15k237_ind']
assert args.lr_scheduler in ['linear', 'cosine']

if not args.train_path and not args.valid_path:
    args.train_path = './data/' + args.task + '/train.txt.json'
    args.valid_path = './data/' + args.task + '/valid.txt.json'
    if args.is_test:
        args.valid_path = './data/' + args.task + '/test.txt.json'


time_str = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

args.model_dir = args.model_dir + args.task.lower() + '-' + time_str + '/'

args.log_path = args.model_dir + 'log.txt'

if args.model_dir:
    os.makedirs(args.model_dir, exist_ok=True)
if args.eval_model_path:
    assert os.path.exists(args.eval_model_path), 'One of args.model_dir and args.eval_model_path should be valid path'
    args.model_dir = os.path.dirname(args.eval_model_path)
    args.log_path = args.model_dir + '/eval_log.txt'

args.seed = 202303
random.seed(args.seed)
torch.manual_seed(args.seed)
cudnn.deterministic = True

try:
    if args.use_amp:
        import torch.cuda.amp
except Exception:
    args.use_amp = False
    warnings.warn('AMP training is not available, set use_amp=False')

if not torch.cuda.is_available():
    args.use_amp = False
    args.print_freq = 1
    warnings.warn('GPU is not available, set use_amp=False and print_freq=1')
