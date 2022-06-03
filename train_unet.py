import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from trainer_unet import trainer_severstal
from config import get_config
import segmentation_models_pytorch as smp

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../dataset', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Severstal', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/list_split/', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=5, help='output channel of network')
parser.add_argument('--output_dir', default='./outputs', type=str, help='output dir')                   
parser.add_argument('--max_iterations', type=int,
                    default=70000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=5, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=12, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=256, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--cfg', type=str, default='configs/swin_tiny_patch4_window7_224_lite.yaml', required=True, metavar="FILE", help='path to config file', )

args = parser.parse_args()
# print("Image Size before Initiation: {}".format(args.img_size))
# if args.dataset == "Severstal":
#     args.root_path = os.path.join(args.root_path)

# config = get_config(args)


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # print("Image Size after Initiation: {}".format(args.img_size))
    dataset_name = args.dataset
    dataset_config = {
        'Severstal': {
            'root_path': args.root_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
        },
    }

    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()

    aux_params=dict(
        dropout=0.2,               # dropout ratio, default is None
        activation='sigmoid',      # activation function, default is None
        classes=5,                 # define number of output labels
    )
    net = smp.Unet('resnet34', classes=5, aux_params=aux_params)
    
    # pretrained_path = config.MODEL.PRETRAIN_CKPT
    # if 'best_model' in pretrained_path:
    #     msg = net.load_state_dict(torch.load(pretrained_path))
    #     print("Using Self Trained swin unet : ",msg)
    # else:
    #     net.load_from(config)

    trainer = {'Severstal': trainer_severstal,}
    trainer[dataset_name](args, net, args.output_dir)