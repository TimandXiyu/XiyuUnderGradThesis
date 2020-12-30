import torch
from torch.autograd import Function
import torch.nn as nn


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


class SoftDiceLoss(nn.Module):
    '''
    Soft_Dice = 2*|dot(A, B)| / (|dot(A, A)| + |dot(B, B)| + eps)
    eps is a small constant to avoid zero division,
    '''

    def __init__(self, weight=None):
        super(SoftDiceLoss, self).__init__()
        self.activation = nn.Softmax2d()

    def forward(self, y_preds, y_truths, eps=1e-8):
        '''
        :param y_preds: [bs,num_classes,768,1024]
        :param y_truths: [bs,num_calsses,768,1024]
        :param eps:
        :return:
        '''
        bs = y_preds.size(0)
        num_classes = y_preds.size(1)
        dices_bs = torch.zeros(bs, num_classes)
        for idx in range(bs):
            y_pred = y_preds[idx]  # [num_classes,768,1024]
            y_truth = y_truths[idx]  # [num_classes,768,1024]
            intersection = torch.sum(torch.mul(y_pred, y_truth), dim=(1, 2)) + eps / 2
            union = torch.sum(torch.mul(y_pred, y_pred), dim=(1, 2)) + torch.sum(torch.mul(y_truth, y_truth),
                                                                                 dim=(1, 2)) + eps

            dices_sub = 2 * intersection / union
            dices_bs[idx] = dices_sub

        dices = torch.mean(dices_bs, dim=0)
        dice = torch.mean(dices)
        dice_loss = 1 - dice
        return dice_loss

class SoftIoULoss(nn.Module):
    '''
    Soft_Dice = 2*|dot(A, B)| / (|dot(A, A)| + |dot(B, B)| + eps)
    eps is a small constant to avoid zero division,
    '''

    def __init__(self, weight=None):
        super(SoftIoULoss, self).__init__()
        self.activation = nn.Softmax2d()

    def forward(self, y_preds, y_truths, eps=1e-8):
        '''
        :param y_preds: [bs,num_classes,768,1024]
        :param y_truths: [bs,num_calsses,768,1024]
        :param eps:
        :return:
        '''
        bs = y_preds.size(0)
        num_classes = y_preds.size(1)
        ious_bs = torch.zeros(bs, num_classes)
        for idx in range(bs):
            y_pred = y_preds[idx]  # [num_classes,768,1024]
            y_truth = y_truths[idx]  # [num_classes,768,1024]
            intersection = torch.sum(torch.mul(y_pred, y_truth), dim=(1, 2)) + eps / 2
            union = torch.sum(torch.mul(y_pred, y_pred), dim=(1, 2)) + torch.sum(torch.mul(y_truth, y_truth),
                                                                                 dim=(1, 2)) + eps

            ious_sub = intersection / (union - intersection)
            ious_bs[idx] = ious_sub

        ious = torch.mean(ious_bs, dim=0)
        iou = torch.mean(ious)
        iou_loss = 1 - iou
        return iou_loss
