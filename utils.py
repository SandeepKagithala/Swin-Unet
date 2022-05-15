import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import os
import torchvision.transforms as T

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        else:
            inputs = torch.sigmoid(inputs)
        
        if inputs.size() != target.size():
            target = self._one_hot_encoder(target)
            
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        preds = torch.argmax(torch.softmax(inputs, dim=1), dim=1)
        preds = self._one_hot_encoder(preds)
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            dice2 = self._dice_loss(preds[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice2.item())
            # class_wise_dice.append(calculate_metric_percase(preds == i, labels == i))
            loss += dice * weight[i]
        mean_dice = sum(class_wise_dice)/self.n_classes
        # mean_dice = np.mean(class_wise_dice, axis=0)[0]
        return loss / self.n_classes, mean_dice

class TverskyLoss(nn.Module):
    def __init__(self, n_classes):
        super(TverskyLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _tversky_loss(self, input, target, alpha=1, beta=1.5):
        target = target.float()
        smooth = 1e-5
        tp = torch.sum(input * target)
        fp = torch.sum((1 - target) * input)
        fn = torch.sum(target * (1 - input))
        tversky = (tp + smooth) / (tp + alpha*fp + beta*fn + smooth)
        loss = 1 - tversky
        return loss

    def _dice_coeff(self, input, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(input * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(input * input)
        dice = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        return dice

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        else:
            inputs = torch.sigmoid(inputs)
        
        if inputs.size() != target.size():
            target = self._one_hot_encoder(target)
            
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        thresholds=[0.4, 0.2, 0.2, 0.3, 0.3]
        if softmax:
            preds = torch.argmax(torch.softmax(inputs, dim=1), dim=1)
            preds = self._one_hot_encoder(preds)
        else:
            for i in range(self.n_classes):
                preds[:, i] = (preds[:, i] > thresholds[i]).astype(np.uint8)
        
        for i in range(0, self.n_classes):
            tversky_loss = self._tversky_loss(inputs[:, i], target[:, i])
            dice = self._dice_coeff(preds[:, i], target[:, i])
            class_wise_dice.append(dice.item())
            loss += tversky_loss * weight[i]
        mean_dice = sum(class_wise_dice)/self.n_classes
        return loss / self.n_classes, mean_dice

def calculate_dice_score(pred, target):
    target = target.astype(np.float32)
    smooth = 1e-5
    intersect = np.sum(pred * target)
    y_sum = np.sum(target * target)
    z_sum = np.sum(pred * pred)
    dice = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    return dice

def one_hot_encoder(input_tensor, num_classes):
        tensor_list = []
        for i in range(num_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0

def test_batch_1(images, labels, net, classes, output_size=[256,1600], test_save_path=None, cases=None):
    # print(output_size)
    # print(output_size.dtype())
    thresholds_max=[0.7, 0.7,0.7,0.7,0.7]
    thresholds_min=[0.4, 0.2,0.2,0.3,0.3]
    min_area=[1000, 500, 500, 750, 1000]
    net.eval()
    images, labels = images.cuda(), labels.cuda()
    with torch.no_grad():
        batch_preds = net(images)
        batch_preds = T.Resize(size=output_size)(batch_preds)
        batch_preds = torch.sigmoid(batch_preds).detach().cpu().numpy()
    
    # labels = T.Resize(size=output_size)(labels)
    labels = one_hot_encoder(labels, classes)
    labels = labels.cpu().detach().numpy()
    x, y = labels.shape[2:]
    labels = zoom(labels, (1, output_size[0]/x, output_size[1]/y), order=0)
    
    predictions = np.zeros_like(labels)
    for k in range(images.shape[0]):
        pred = batch_preds[k]
        fname = cases[k]
        gt_label = labels[k]
        pred_masks = []
        for i in range(classes):
            p_channel = pred[i]
            p_channel_ = p_channel
            p_channel = (p_channel>thresholds_max[i]).astype(np.uint8)
            if p_channel.sum() < min_area[i]:
                p_channel = np.zeros(p_channel.shape, dtype=p_channel.dtype)
            else:
                p_channel = (p_channel_>thresholds_min[i]).astype(np.uint8)
            pred_masks.append(p_channel)
        pred_masks = np.array(pred_masks)
        predictions[k] = pred_masks
        
        if test_save_path is not None:
            np.savez_compressed(os.path.join(test_save_path, fname + "_pred.npz"), label = gt_label, pred = pred_masks)


    metric_list = []
    for i in range(classes):
        # metric_list.append(calculate_metric_percase(predictions[:,i], labels[:,i])) 
        metric_list.append(calculate_dice_score(predictions[:,i], labels[:,i])) 

    return metric_list
           

def test_batch(images, labels, net, classes, output_size=[256, 1600], test_save_path=None, cases=None):
    
    images, labels = images.cuda(), labels.cuda()
    with torch.no_grad():
        batch_preds = net(images)
        predictions = torch.argmax(torch.softmax(batch_preds, dim=1), dim=1)
    
    predictions = predictions.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    x,y = predictions.shape[1:]
    predictions = zoom(predictions, (1, output_size[0]/x, output_size[1]/y), order=0)
    labels = zoom(labels, (1, output_size[0]/x, output_size[1]/y), order=0)
    
    if test_save_path is not None:
        for k in range(images.shape[0]):
            pred = predictions[k]
            fname = cases[k]
            gt_label = labels[k]
            np.savez_compressed(os.path.join(test_save_path, fname + "_pred.npz"), label = gt_label, pred = pred)

    metric_list = []
    for i in range(1, classes):
        # metric_list.append(calculate_metric_percase(predictions == i, labels == i))  
        metric_list.append(calculate_dice_score(predictions == i, labels == i))

    return metric_list


def test_single_batch(image, label, net, classes, patch_size=[224, 224], test_save_path=None, cases=None):
    images, labels = image.cpu().detach().numpy(), label.cpu().detach().numpy()
    predictions = np.zeros_like(labels)
    for ind in range(images.shape[0]):
        img = images[ind, :, :]
        mask = labels[ind, :, :]
        case_name = cases[ind]
        x, y = img.shape[0], img.shape[1]
        if x != patch_size[0] or y != patch_size[1]:
            img = zoom(img, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
        input = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            outputs = net(input)
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            if x != patch_size[0] or y != patch_size[1]:
                pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            else:
                pred = out
            predictions[ind] = pred

        if test_save_path is not None:
            np.savez_compressed(os.path.join(test_save_path, case_name + "_pred.npz"), image = images[ind, :, :].astype(np.float32), label = mask, pred = pred)
    
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(predictions == i, labels == i))

    return metric_list


def test_single_volume(image, label, net, classes, patch_size=[224, 224], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list