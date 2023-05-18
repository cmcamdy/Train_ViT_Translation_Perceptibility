from utils.mix import cutmix_data, mixup_data, mixup_criterion
import numpy as np
import random
import logging as log
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from colorama import Fore, Style
from torchsummary import summary
from utils.losses import LabelSmoothingCrossEntropy
import os
from utils.sampler import RASampler
# import models.create_model as m
from utils.logger_dict import Logger_dict
from utils.print_progress import progress_bar
from utils.training_functions import accuracy
import argparse
from utils.scheduler import build_scheduler
from utils.dataloader import datainfo, dataload
# from models.create_model import create_model
from models.build_model_affine_sep import create_model
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=Warning)
from utils.translation_policy import TranslatePolicy

best_acc1 = 0
MODELS = ['vit', 'swin', 'pit', 'cait', 't2t']


def _weight_decay(init_weight, epoch, warmup_epochs=20, total_epoch=300):
    # import pdb; pdb.set_trace()
    if epoch > total_epoch * 0.9:
        cur_weight = 0
    elif epoch <= warmup_epochs:
        cur_weight = min(init_weight / warmup_epochs * epoch, init_weight)
    else:
        cur_weight = init_weight * (1.0 - (epoch - warmup_epochs) /
                                    (total_epoch - warmup_epochs))
    return cur_weight


def init_parser():
    parser = argparse.ArgumentParser(
        description='CIFAR10 quick training script')

    # Data args
    parser.add_argument('--data_path',
                        default='./dataset',
                        type=str,
                        help='dataset path')

    parser.add_argument(
        '--dataset',
        default='CIFAR10',
        choices=['CIFAR10', 'CIFAR100', 'T-IMNET', 'T-IMNET-100', 'SVHN'],
        type=str,
        help='Image Net dataset path')

    parser.add_argument('-j',
                        '--workers',
                        default=4,
                        type=int,
                        metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--print-freq',
                        default=20,
                        type=int,
                        metavar='N',
                        help='log frequency (by iteration)')

    # Optimization hyperparams
    parser.add_argument('--epochs',
                        default=300,
                        type=int,
                        metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--warmup',
                        default=10,
                        type=int,
                        metavar='N',
                        help='number of warmup epochs')

    parser.add_argument('-b',
                        '--batch_size',
                        default=128,
                        type=int,
                        metavar='N',
                        help='mini-batch size (default: 128)',
                        dest='batch_size')

    parser.add_argument('--lr',
                        default=0.001,
                        type=float,
                        help='initial learning rate')

    parser.add_argument('--weight-decay',
                        default=5e-2,
                        type=float,
                        help='weight decay (default: 1e-4)')

    parser.add_argument('--model', type=str, default='vit', choices=MODELS)

    parser.add_argument('--disable-cos',
                        action='store_true',
                        help='disable cosine lr schedule')

    parser.add_argument('--enable_aug',
                        action='store_true',
                        help='disable augmentation policies for training')

    parser.add_argument('--gpu', default=0, type=int)

    parser.add_argument('--no_cuda', action='store_true', help='disable cuda')

    parser.add_argument('--ls', action='store_false', help='label smoothing')

    parser.add_argument('--channel', type=int, help='disable cuda')

    parser.add_argument('--heads', type=int, help='disable cuda')

    parser.add_argument('--depth', type=int, help='disable cuda')

    parser.add_argument('--tag', type=str, help='tag', default='')

    parser.add_argument('--seed', type=int, default=0, help='seed')

    parser.add_argument('--sd',
                        default=0.1,
                        type=float,
                        help='rate of stochastic depth')

    parser.add_argument('--resume', default=False, help='Version')

    parser.add_argument('--aa',
                        action='store_false',
                        help='Auto augmentation used'),

    parser.add_argument('--smoothing',
                        type=float,
                        default=0.1,
                        help='Label smoothing (default: 0.1)')

    parser.add_argument('--cm', action='store_false', help='Use Cutmix')

    parser.add_argument('--beta',
                        default=1.0,
                        type=float,
                        help='hyperparameter beta (default: 1)')

    parser.add_argument('--mu', action='store_false', help='Use Mixup')

    parser.add_argument('--alpha',
                        default=1.0,
                        type=float,
                        help='mixup interpolation coefficient (default: 1)')

    parser.add_argument('--mix_prob',
                        default=0.5,
                        type=float,
                        help='mixup probability')

    parser.add_argument('--ra',
                        type=int,
                        default=3,
                        help='repeated augmentation')

    parser.add_argument('--re',
                        default=0.25,
                        type=float,
                        help='Random Erasing probability')

    parser.add_argument('--re_sh',
                        default=0.4,
                        type=float,
                        help='max erasing area')

    parser.add_argument('--re_r1',
                        default=0.3,
                        type=float,
                        help='aspect of erasing area')

    parser.add_argument('--is_LSA',
                        action='store_true',
                        help='Locality Self-Attention')

    parser.add_argument('--is_SPT',
                        action='store_true',
                        help='Shifted Patch Tokenization')

    parser.add_argument(
        '--save_path',
        default='/work/workspace12/chenhuan/code/outputs/spt_lsa',
        type=str,
        help='dataset path')

    parser.add_argument('--alpha_trans',
                        default=1.0,
                        type=float,
                        help='alpha_trans')
    parser.add_argument('--magnitude',
                        default=0.3,
                        type=float,
                        help='magnitude')
    parser.add_argument('--with_scale', action="store_true")
    parser.add_argument('--with_trans', action="store_true")
    parser.add_argument('--scale', default=0.5, type=float, help='magnitude')
    parser.add_argument('--init_weight',
                        default=1.0,
                        type=float,
                        help='magnitude')
    return parser


def main(args):
    global best_acc1

    torch.cuda.set_device(args.gpu)

    data_info = datainfo(logger, args)

    model = create_model(data_info['img_size'], data_info['n_classes'], args)

    model.cuda(args.gpu)

    print(Fore.GREEN + '*' * 80)
    logger.debug(f"Creating model: {model_name}")
    n_parameters = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
    logger.debug(f'Number of params: {format(n_parameters, ",")}')
    logger.debug(f'Initial learning rate: {args.lr:.6f}')
    logger.debug(f"Start training for {args.epochs} epochs")
    print('*' * 80 + Style.RESET_ALL)

    if args.ls:
        print(Fore.YELLOW + '*' * 80)
        logger.debug('label smoothing used')
        print('*' * 80 + Style.RESET_ALL)
        criterion = LabelSmoothingCrossEntropy()

    else:
        criterion = nn.CrossEntropyLoss()

    if args.sd > 0.:
        print(Fore.YELLOW + '*' * 80)
        logger.debug(f'Stochastic depth({args.sd}) used ')
        print('*' * 80 + Style.RESET_ALL)

    criterion = criterion.cuda(args.gpu)
    criterion_trans = torch.nn.L1Loss()
    criterion_scale = torch.nn.L1Loss()

    normalize = [
        transforms.Normalize(mean=data_info['stat'][0],
                             std=data_info['stat'][1])
    ]

    if args.cm:
        print(Fore.YELLOW + '*' * 80)
        logger.debug('Cutmix used')
        print('*' * 80 + Style.RESET_ALL)
    if args.mu:
        print(Fore.YELLOW + '*' * 80)
        logger.debug('Mixup used')
        print('*' * 80 + Style.RESET_ALL)
    if args.ra > 1:

        print(Fore.YELLOW + '*' * 80)
        logger.debug(f'Repeated Aug({args.ra}) used')
        print('*' * 80 + Style.RESET_ALL)
    '''
        Data Augmentation
    '''
    augmentations = []

    augmentations += [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(data_info['img_size'], padding=4)
    ]

    if args.aa == True:
        print(Fore.YELLOW + '*' * 80)
        logger.debug('Autoaugmentation used')

        if 'CIFAR' in args.dataset:
            print("CIFAR Policy")
            from utils.autoaug import CIFAR10Policy
            augmentations += [CIFAR10Policy()]

        elif 'SVHN' in args.dataset:
            print("SVHN Policy")
            from utils.autoaug import SVHNPolicy
            augmentations += [SVHNPolicy()]

        else:
            from utils.autoaug import ImageNetPolicy
            augmentations += [ImageNetPolicy()]

        print('*' * 80 + Style.RESET_ALL)

    augmentations += [transforms.ToTensor(), *normalize]

    if args.re > 0:
        from utils.random_erasing import RandomErasing
        print(Fore.YELLOW + '*' * 80)
        logger.debug(f'Random erasing({args.re}) used ')
        print('*' * 80 + Style.RESET_ALL)

        augmentations += [
            RandomErasing(probability=args.re,
                          sh=args.re_sh,
                          r1=args.re_r1,
                          mean=data_info['stat'][0])
        ]

    augmentations = transforms.Compose(augmentations)
    trans = TranslatePolicy(magnitude=np.linspace(-args.magnitude,
                                                  args.magnitude, 10),
                            scale=np.linspace(1.0 - args.scale,
                                              1.0 + args.scale, 10),
                            with_scale=args.with_scale,
                            with_trans=args.with_trans)

    train_dataset, val_dataset = dataload(args, augmentations, normalize,
                                          data_info)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               batch_sampler=RASampler(
                                                   len(train_dataset),
                                                   args.batch_size,
                                                   1,
                                                   args.ra,
                                                   shuffle=True,
                                                   drop_last=True))
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=args.workers)
    '''
        Training
    '''

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = build_scheduler(args, optimizer, len(train_loader))

    summary(model.backbone, (3, data_info['img_size'], data_info['img_size']))

    print()
    print("Beginning training")
    print()

    lr = optimizer.param_groups[0]["lr"]

    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        final_epoch = args.epochs
        args.epochs = final_epoch - (checkpoint['epoch'] + 1)

    for epoch in tqdm(range(args.epochs)):
        lambda_trans = _weight_decay(args.init_weight, epoch,
                                     args.epochs * 0.2, args.epochs)
        lr = train(train_loader, model, criterion, criterion_trans,
                   criterion_scale, lambda_trans, trans, optimizer, epoch,
                   scheduler, args)
        acc1 = validate(
            val_loader,
            model.backbone if args.with_scale or args.with_trans else model,
            criterion,
            lr,
            args,
            epoch=epoch)
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, os.path.join(save_path, 'checkpoint.pth'))

        logger_dict.print()

        if acc1 > best_acc1:
            print('* Best model upate *')
            best_acc1 = acc1

            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, os.path.join(save_path, 'best.pth'))

        print(f'Best acc1 {best_acc1:.2f}')
        print('*' * 80)
        print(Style.RESET_ALL)

        writer.add_scalar("Learning Rate", lr, epoch)

    print(Fore.RED + '*' * 80)
    logger.debug(f'best top-1: {best_acc1:.2f}, final top-1: {acc1:.2f}')
    print('*' * 80 + Style.RESET_ALL)
    torch.save(model.state_dict(), os.path.join(save_path, 'checkpoint.pth'))


def affine_infer(model,
                 images,
                 targets,
                 criterion,
                 criterion_trans,
                 criterion_scale,
                 transpolicy,
                 lambda_trans,
                 y_a=None,
                 y_b=None,
                 lam=None,
                 args=None):
    loss_oral, loss_translated, loss_trans_mse, loss_scale_mse = None, 0.0, 0.0, 0.0
    outputs, outputs_translated, trans_dis, scale_dis = None, None, None, None
    # affine
    images_translated, targets_translate = transpolicy(images)

    # infer
    if args.with_trans and not args.with_scale:
        outputs, outputs_translated, trans_dis = model(images,
                                                       images_translated)
    elif not args.with_trans and args.with_scale:
        outputs, outputs_translated, scale_dis = model(images,
                                                       images_translated)
    elif not args.with_trans and not args.with_scale:
        # outputs, _ = model.backbone(images)
        outputs, _ = model(images)
    else:
        outputs, outputs_translated, trans_dis, scale_dis = model(
            images, images_translated)

    # cal loss
    if y_a is None and y_b is None and lam is None:
        loss_oral = criterion(outputs, targets)
        if args.with_trans or args.with_scale:
            loss_translated = criterion(outputs_translated, targets)
    else:
        loss_oral = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        if args.with_trans or args.with_scale:
            loss_translated = mixup_criterion(criterion, outputs_translated,
                                              y_a, y_b, lam)
    # import pdb; pdb.set_trace()
    if trans_dis is not None:
        if scale_dis is None:
            targets_trans = targets_translate
        else:
            # import pdb; pdb.set_trace()
            targets_trans = targets_translate[:, :2]
        loss_trans_mse = criterion_trans(trans_dis, targets_trans)
    else:
        loss_trans_mse = 0

    if scale_dis is not None:
        if trans_dis is None:
            targets_scale = targets_translate
        else:
            # import pdb; pdb.set_trace()
            targets_scale = targets_translate[:, 2:]
        loss_scale_mse = criterion_scale(scale_dis, targets_scale)
    else:
        loss_scale_mse = 0
    # v8的改动：需要凸显原先loss的重要性
    # import pdb; pdb.set_trace()
    if args.with_scale and not args.with_trans:
        loss = loss_translated * args.alpha_trans + loss_scale_mse * lambda_trans
        loss = loss / 2 + loss_oral
    elif not args.with_scale and args.with_trans:
        loss = loss_translated * args.alpha_trans + loss_trans_mse * lambda_trans
        loss = loss / 2 + loss_oral
    elif args.with_scale and args.with_trans:
        loss = loss_translated * args.alpha_trans + loss_scale_mse + loss_trans_mse * lambda_trans
        loss = loss / 3 + loss_oral
    else:
        loss = loss_oral
    # print(
    #     "loss:{}, loss_oral:{}, loss_translated:{}, loss_scale_mse:{}, loss_trans_mse:{}"
    #     .format(loss, loss_oral, loss_translated * args.alpha_trans, loss_scale_mse * lambda_trans, loss_trans_mse * lambda_trans))
    return loss, outputs


def train(train_loader, model, criterion, criterion_trans, criterion_scale,
          lambda_trans, transpolicy, optimizer, epoch, scheduler, args):
    model.train()
    loss_val, acc1_val = 0, 0
    n = 0

    for i, (images, target) in enumerate(train_loader):
        if (not args.no_cuda) and torch.cuda.is_available():
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        # Cutmix only
        if args.cm and not args.mu:
            r = np.random.rand(1)
            if r < args.mix_prob:
                slicing_idx, y_a, y_b, lam, sliced = cutmix_data(
                    images, target, args)
                images[:, :, slicing_idx[0]:slicing_idx[2],
                       slicing_idx[1]:slicing_idx[3]] = sliced
                loss, output = affine_infer(model,
                                            images,
                                            target,
                                            criterion,
                                            criterion_trans,
                                            criterion_scale,
                                            transpolicy,
                                            lambda_trans,
                                            y_a,
                                            y_b,
                                            lam,
                                            args=args)

            else:
                loss, output = affine_infer(model,
                                            images,
                                            target,
                                            criterion,
                                            criterion_trans,
                                            criterion_scale,
                                            transpolicy,
                                            lambda_trans,
                                            args=args)

        # Mixup only
        elif not args.cm and args.mu:
            r = np.random.rand(1)
            if r < args.mix_prob:
                images, y_a, y_b, lam = mixup_data(images, target, args)
                loss, output = affine_infer(model,
                                            images,
                                            target,
                                            criterion,
                                            criterion_trans,
                                            criterion_scale,
                                            transpolicy,
                                            lambda_trans,
                                            y_a,
                                            y_b,
                                            lam,
                                            args=args)

            else:
                loss, output = affine_infer(model,
                                            images,
                                            target,
                                            criterion,
                                            criterion_trans,
                                            criterion_scale,
                                            transpolicy,
                                            lambda_trans,
                                            args=args)

        # Both Cutmix and Mixup
        elif args.cm and args.mu:
            r = np.random.rand(1)
            if r < args.mix_prob:
                switching_prob = np.random.rand(1)

                # Cutmix
                if switching_prob < 0.5:
                    slicing_idx, y_a, y_b, lam, sliced = cutmix_data(
                        images, target, args)
                    images[:, :, slicing_idx[0]:slicing_idx[2],
                           slicing_idx[1]:slicing_idx[3]] = sliced
                    loss, output = affine_infer(model,
                                                images,
                                                target,
                                                criterion,
                                                criterion_trans,
                                                criterion_scale,
                                                transpolicy,
                                                lambda_trans,
                                                y_a,
                                                y_b,
                                                lam,
                                                args=args)

                # Mixup
                else:
                    images, y_a, y_b, lam = mixup_data(images, target, args)
                    loss, output = affine_infer(model,
                                                images,
                                                target,
                                                criterion,
                                                criterion_trans,
                                                criterion_scale,
                                                transpolicy,
                                                lambda_trans,
                                                y_a,
                                                y_b,
                                                lam,
                                                args=args)

            else:
                loss, output = affine_infer(model,
                                            images,
                                            target,
                                            criterion,
                                            criterion_trans,
                                            criterion_scale,
                                            transpolicy,
                                            lambda_trans,
                                            args=args)

        # No Mix
        else:
            loss, output = affine_infer(model,
                                        images,
                                        target,
                                        criterion,
                                        criterion_trans,
                                        criterion_scale,
                                        transpolicy,
                                        lambda_trans,
                                        args=args)

        acc = accuracy(output, target, (1, ))
        acc1 = acc[0]
        n += images.size(0)
        loss_val += float(loss.item() * images.size(0))
        acc1_val += float(acc1[0] * images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        if args.print_freq >= 0 and i % args.print_freq == 0:
            avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
            progress_bar(
                i, len(train_loader),
                f'[Epoch {epoch+1}/{args.epochs}][T][{i}]   Loss: {avg_loss:.4e}   Top-1: {avg_acc1:6.2f}   LR: {lr:.7f}'
                + ' ' * 10)

    logger_dict.update(keys[0], avg_loss)
    logger_dict.update(keys[1], avg_acc1)
    writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.add_scalar("Acc/train", avg_acc1, epoch)

    return lr


def validate(val_loader, model, criterion, lr, args, epoch=None):
    model.eval()
    loss_val, acc1_val = 0, 0
    n = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if (not args.no_cuda) and torch.cuda.is_available():
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            output, _ = model(images)
            loss = criterion(output, target)

            acc = accuracy(output, target, (1, 5))
            acc1 = acc[0]
            n += images.size(0)
            loss_val += float(loss.item() * images.size(0))
            acc1_val += float(acc1[0] * images.size(0))

            if args.print_freq >= 0 and i % args.print_freq == 0:
                avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
                progress_bar(
                    i, len(val_loader),
                    f'[Epoch {epoch+1}][V][{i}]   Loss: {avg_loss:.4e}   Top-1: {avg_acc1:6.2f}   LR: {lr:.6f}'
                )
    print()

    print(Fore.BLUE)
    print('*' * 80)

    logger_dict.update(keys[2], avg_loss)
    logger_dict.update(keys[3], avg_acc1)

    writer.add_scalar("Loss/val", avg_loss, epoch)
    writer.add_scalar("Acc/val", avg_acc1, epoch)

    return avg_acc1


if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()
    global save_path
    global writer

    # random seed

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    np.random.seed(args.seed)  # Numpy module.
    random.seed(args.seed)  # Python random module.
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    model_name = args.model

    if not args.is_SPT:
        model_name += "-Base"
    else:
        model_name += "-SPT"

    if args.is_LSA:
        model_name += "-LSA"

    model_name += f"-{args.tag}-{args.dataset}-LR[{args.lr}]-Seed{args.seed}"
    save_path = os.path.join(args.save_path, 'save', model_name)
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    writer = SummaryWriter(os.path.join(os.getcwd(), 'tensorboard',
                                        model_name))

    # logger

    log_dir = os.path.join(save_path, 'history.csv')
    logger = log.getLogger(__name__)
    formatter = log.Formatter('%(message)s')
    streamHandler = log.StreamHandler()
    fileHandler = log.FileHandler(log_dir, 'a')
    streamHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)
    logger.setLevel(level=log.DEBUG)

    global logger_dict
    global keys

    logger_dict = Logger_dict(logger, save_path)
    keys = ['T Loss', 'T Top-1', 'V Loss', 'V Top-1']

    main(args)
