import argparse
import logging
import os
import sys

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
from utils.dice_loss import SoftDiceLoss
import torch.backends.cudnn
from unet.dinknet import DinkNet34 as DlinkNet34
from unet.dinknet import DinkNet101 as DlinkNet101
from unet.dinknet import DinkNet50 as DlinkNet50

dir_img = r'./data/mixed_data_3.0/'
dir_mask = r'./data/mixed_mask_3.0/'
dir_checkpoint = r'checkpoints/'

cross_dir_img = r'./data/cz/original_content/'
cross_dir_mask = r'./data/cz/original_mask/'


def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5,
              val_ignore_index=None):
    torch.manual_seed(1234)
    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    cross_dataset = BasicDataset(cross_dir_img, cross_dir_mask, img_scale)
    cross_dataset.aug = False
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    if val_ignore_index is not None:
        val_indeces = val.indices
        reference = val_indeces.copy()
        for i, ele in enumerate(reference):
            if ele > val_ignore_index:
                val_indeces.remove(ele)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    cross_val_loader = DataLoader(cross_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
                                  drop_last=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    # optimizer.load_state_dict(torch.load(r'.\checkpoints\CP_Optimizer_32.pth'))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max',
                                                     factor=0.6,
                                                     patience=5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=5)
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        # criterion2 = SoftIoULoss()
        criterion1 = nn.BCELoss()
        criterion2 = SoftDiceLoss()
        # criterion = focal_loss()
        # criterion = nn.NLLLoss()

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                dataset.aug = True
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long

                true_masks = true_masks.to(device=device, dtype=mask_type)

                true_masks = true_masks.float()

                masks_pred = net(imgs)
                if net.n_classes == 1:
                    loss = criterion2(masks_pred, true_masks) + criterion1(masks_pred, true_masks)
                else:
                    loss = criterion(masks_pred, true_masks)

                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item(), 'learning rate': optimizer.param_groups[0]['lr']})

                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (2 * n_train // batch_size) == 0:
                    cross_val_score = eval_net(net, cross_val_loader, device)
                    writer.add_scalar('Cross_Dice/test', cross_val_score, global_step)
                    logging.info('Cross Validation Dict/test', cross_val_score)

                if global_step % (n_train // (2 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    dataset.aug = False
                    val_score = eval_net(net, val_loader, device)
                    scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test', val_score, global_step)

                    writer.add_images('images', imgs, global_step)
                    if net.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', masks_pred > 0.5, global_step)

        if save_cp and global_step % (4 * n_train // batch_size) == 0:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            torch.save(optimizer.state_dict(),
                       dir_checkpoint + f'CP_Optimizer_{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    args.epochs = 100
    args.batchsize = 2
    args.scale = [1024, 1024]
    args.lr = 1e-5
    args.val = 10
    args.load = r'./checkpoints/CP_epoch28.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = DlinkNet101(num_classes=1, num_channels=3)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )

        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    torch.backends.cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  val_ignore_index=6225)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)