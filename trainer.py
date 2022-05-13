import argparse
import logging
import os
import random
import sys
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss, TverskyLoss
from torchvision import transforms
from utils import test_single_batch
from dataset_severstal import Severstal_dataset, RandomGenerator

def trainer_severstal(args, model, snapshot_path):
    
    logging.basicConfig(filename=snapshot_path + "/log_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".log", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    start_time = time.time()

    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Severstal_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train", output_size=(args.img_size, args.img_size))
    db_val = Severstal_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="val", output_size=(args.img_size, args.img_size))

    print("The length of train set is: {}".format(len(db_train)))
    print("The length of validation set is: {}".format(len(db_val)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    #trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8, worker_init_fn=worker_init_fn)
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers = 2)
    valloader = DataLoader(db_val, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers = 2)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    ce_loss = CrossEntropyLoss()
    bce_loss = BCEWithLogitsLoss()
    dice_loss = DiceLoss(num_classes)
    tversky_loss = TverskyLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_mean_dice = 0
    weights=[1, 1.4, 1, 1.4, 1]
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        # Model Training
        model.train()
        train_loss_sum = 0
        train_loss_ce_sum = 0
        train_mean_dice_sum = 0
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            with torch.set_grad_enabled(True):
                outputs = model(image_batch)
                # loss_ce = bce_loss(outputs, label_batch)
                loss_ce = ce_loss(outputs, label_batch[:].long())
                # loss_dice, mean_dice = dice_loss(outputs, label_batch, weight=weights, softmax=True)
                # loss = 0.3 * loss_ce + 0.7 * loss_dice
                loss_tversky, mean_dice = tversky_loss(outputs, label_batch, weights, softmax=True)
                loss = 0.4 * loss_ce + 0.6 * loss_tversky
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            train_loss_sum += loss
            train_loss_ce_sum += loss_ce
            train_mean_dice_sum += mean_dice
            # logging.info('iteration %d : loss : %f, loss_ce: %f, mean_dice: %f' % (iter_num, loss.item(), loss_ce.item(), mean_dice))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                # outputs = torch.sigmoid(outputs)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        avg_train_loss = train_loss_sum / len(trainloader)
        avg_train_loss_ce = train_loss_ce_sum / len(trainloader)
        avg_train_mean_dice = train_mean_dice_sum / len(trainloader)

        writer.add_scalar('info/train_total_loss', avg_train_loss, epoch_num)
        writer.add_scalar('info/train_loss_ce', avg_train_loss_ce, epoch_num)
        writer.add_scalar('info/train_mean_dice', avg_train_mean_dice, epoch_num)
        logging.info("Average Training Loss: {}, Average Mean Dice Score: {}".format(avg_train_loss, avg_train_mean_dice))

        #Validation Testing
        model.eval()
        val_loss_sum = 0
        val_loss_ce_sum = 0
        val_mean_dice_sum = 0
        for i_batch, sampled_batch in enumerate(valloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            with torch.no_grad():
                outputs = model(image_batch)
                # loss_ce = bce_loss(outputs, label_batch)
                loss_ce = ce_loss(outputs, label_batch[:].long())
                loss_dice, mean_dice = dice_loss(outputs, label_batch, softmax=True)
                loss = 0.3 * loss_ce + 0.7 * loss_dice
                val_loss_sum += loss
                val_loss_ce_sum += loss_ce
                val_mean_dice_sum += mean_dice

        avg_loss = val_loss_sum / len(valloader)
        avg_loss_ce = val_loss_ce_sum / len(valloader)
        avg_mean_dice = val_mean_dice_sum / len(valloader)

        writer.add_scalar('info/val_total_loss', avg_loss, epoch_num)
        writer.add_scalar('info/val_loss_ce', avg_loss_ce, epoch_num)
        writer.add_scalar('info/val_mean_dice', avg_mean_dice, epoch_num)
        logging.info("Average Validation Loss: {}, Average Mean Dice Score: {}".format(avg_loss, avg_mean_dice))

        if avg_mean_dice > best_mean_dice:
            best_mean_dice = avg_mean_dice
            if epoch_num > 10:
                logging.info("Best Dice Score found at epoch {}".format(epoch_num + 1))
                save_mode_path = os.path.join(snapshot_path, 'best_model' + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("Best Model saved to {}".format(save_mode_path))


        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        #Saving model at final epoch
        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break
    
    elapsed_time = time.time() - start_time
    logging.info("Training Time in Minutes : {}".format(round(elapsed_time/60.0, 2)))
    writer.close()
    return "Training Finished!"