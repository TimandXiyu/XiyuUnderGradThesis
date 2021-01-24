import logging
import os
import sys

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader
import torch.backends.cudnn
from unet.dinknet import DinkNet101 as DlinkNet101
from unet.dinknet import DinkNet34 as DlinkNet34
from unet.dinknet import DinkNet50 as DlinkNet50
from test import test_net


dir_img = './data/cz/cropped_cz_src/'
dir_mask = './data/cz/cropped_cz_mask/'


def tst_net(net,
            device,
            batch_size=1,
            img_scale=0.5):

    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    tst_size = len(dataset)
    tst_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    dataset.aug = False

    writer = SummaryWriter(comment=f'BS_{batch_size}_SCALE_{img_scale}_TEST_RUN')
    global_step = 0

    logging.info(f'''Starting training:
        Batch size:      {batch_size}
        Test size        {tst_size}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    tst_score, imgs, masks_pred, true_masks = test_net(net, tst_loader, device)

    if net.n_classes > 1:
        logging.info('Validation cross entropy: {}'.format(tst_score))
        writer.add_scalar('Loss/test', tst_score, global_step)
    else:

        logging.info('Validation Dice Coeff: {}'.format(tst_score))
        writer.add_scalar('Dice/test', tst_score, global_step)

    writer.add_images('images', imgs, global_step)
    if net.n_classes == 1:
        writer.add_images('masks/true', true_masks, global_step)
        writer.add_images('masks/pred', masks_pred > 0.5, global_step)

    writer.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    batchsize = 2
    scale = [1024, 1024]
    load_dir = r'D:\NetworkCheckpoints\CP Dlinknet101 auged 3.0\CP_epoch28.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = DlinkNet101(num_classes=1, num_channels=3)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n')

    if load_dir:
        net.load_state_dict(
            torch.load(load_dir, map_location=device)
        )
        logging.info(f'Model loaded from {load_dir}')

    net.to(device=device)
    # faster convolutions, but more memory
    torch.backends.cudnn.benchmark = True

    try:
        tst_net(net=net,
                batch_size=batchsize,
                device=device,
                img_scale=scale,)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)