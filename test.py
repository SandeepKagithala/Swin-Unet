import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset_severstal import Severstal_dataset
from utils import test_single_batch, test_batch, test_batch_1
from networks.vision_transformer import SwinUnet as ViT_seg
from trainer import trainer_severstal
from config import get_config

parser = argparse.ArgumentParser()
parser.add_argument('--test_images_dir', type=str,
                    default='../dataset', help='root dir for test Image data')  
parser.add_argument('--dataset', type=str,
                    default='Severstal', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=5, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists', help='list dir')
parser.add_argument('--output_dir', type=str, help='output dir')   
parser.add_argument('--max_iterations', type=int,default=50000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--output_size', type=int, nargs='+', default=[256,1600], help='output size of network output')
parser.add_argument('--is_save', action="store_true", help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args()
# if args.dataset == "Severstal":
#     args.test_images_dir = os.path.join(args.test_images_dir, "test_vol_h5")
config = get_config(args)


def inference(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.test_images_dir, list_dir=args.list_dir, split="test", output_size=(args.img_size, args.img_size))
    testloader = DataLoader(db_test, batch_size=24, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        images, labels, case_names = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name']
        metric_i = test_batch_1(images, labels, model, classes=args.num_classes, output_size=[args.output_size[0], args.output_size[1]],
                                      test_save_path=test_save_path, cases=case_names)
        metric_list += np.array(metric_i)
        # logging.info('batch_idx %d mean_dice %f mean_hd95 %f' % (i_batch, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
        logging.info('batch_idx %d mean_dice %f' % (i_batch, np.mean(metric_i)))
    metric_list = metric_list / len(testloader)
    for i in range(1, args.num_classes):
        # logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
        logging.info('Mean class %d mean_dice %f ' % (i, metric_list[i-1]))
    # performance = np.mean(metric_list, axis=0)[0]
    # mean_hd95 = np.mean(metric_list, axis=0)[1]
    # logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    performance = np.mean(metric_list)
    logging.info('Testing performance in best val model: mean_dice : %f ' % (performance))
    return "Testing Finished!"


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

    dataset_config = {
        'Severstal': {
            'Dataset': Severstal_dataset,
            'test_images_dir': args.test_images_dir,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.test_images_dir = dataset_config[dataset_name]['test_images_dir']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True

    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()

    snapshot = os.path.join(args.output_dir, 'best_model.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
    msg = net.load_state_dict(torch.load(snapshot))
    print("self trained swin unet",msg)
    snapshot_name = os.path.basename(snapshot)

    log_folder = './test_log/test_log_'
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_folder, snapshot_name+".txt"), level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_save:
        args.test_save_dir = os.path.join(args.output_dir, "predictions")
        test_save_path = args.test_save_dir 
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)


