import argparse
from collections import OrderedDict
import datetime
import json
import os
import time

from einops import rearrange
import kornia as K
import torch
import torch.cuda.amp as amp
from torch.distributed.optim import ZeroRedundancyOptimizer
import torchvision
from timm.data.loader import MultiEpochsDataLoader

from avion.data.kinetics_dataset import KineticsDataset
from avion.data.transforms import GroupMultiScaleCrop, Permute, TubeMaskingGeneratorGPU
import avion.models.model_videomae as model_videomae
from avion.optim.lion import Lion
from avion.optim.schedulers import cosine_scheduler
import avion.utils.distributed as dist_utils
from avion.utils.meters import AverageMeter, ProgressMeter
from avion.utils.misc import check_loss_nan


def get_args_parser():
    parser = argparse.ArgumentParser(description='VideoMAE pretrain', add_help=False)
    parser.add_argument('--root',
                        default='datasets/Kinetics/train_320px/',
                        type=str, help='path to train dataset root')
    parser.add_argument('--train-metadata',
                        default='datasets/Kinetics/annotations/k400_train_list.txt',
                        type=str, help='metadata for train split')
    parser.add_argument('--output-dir', default='./', type=str, help='output dir')
    parser.add_argument('--input-size', default=224, type=int, help='input frame size')
    parser.add_argument('--clip-length', default=16, type=int, help='clip length')
    parser.add_argument('--clip-stride', default=4, type=int, help='clip stride')
    parser.add_argument('--mask-ratio', default=0.9, type=float, help='mask ratio')
    parser.add_argument('--fused-decode-crop', action='store_true', dest='fused_decode_crop')
    parser.add_argument('--no-fused-decode-crop', action='store_false', dest='fused_decode_crop')
    parser.set_defaults(fused_decode_crop=False)
    parser.add_argument('--decode-threads', default=1, type=int)
    parser.add_argument('--use-pin-memory', action='store_true', dest='use_pin_memory')
    parser.add_argument('--disable-pin-memory', action='store_false', dest='use_pin_memory')
    parser.set_defaults(use_pin_memory=False)
    # model
    parser.add_argument('--model', default='VIDEOMAE_VITB16', type=str)
    parser.add_argument('--channel-last', action='store_true', dest='channel_last')
    parser.add_argument('--disable-channel-last', action='store_false', dest='channel_last')
    parser.set_defaults(channel_last=False)
    parser.add_argument('--decoder-depth', default=4, type=int, help='decoder depth')
    parser.add_argument('--grad-checkpointing', action='store_true', dest='use_grad_checkpointing')
    parser.add_argument('--no-grad-checkpointing', action='store_false', dest='use_grad_checkpointing')
    parser.set_defaults(use_grad_checkpointing=False)
    parser.add_argument('--use-flash-attn-at-encoder', action='store_true', dest='use_flash_attn_at_encoder')
    parser.add_argument('--disable-flash-attn-at-encoder', action='store_false', dest='use_flash_attn_at_encoder')
    parser.set_defaults(use_flash_attn_at_encoder=False)
    parser.add_argument('--use-flash-attn-at-decoder', action='store_true', dest='use_flash_attn_at_decoder')
    parser.add_argument('--disable-flash-attn-at-decoder', action='store_false', dest='use_flash_attn_at_decoder')
    parser.set_defaults(use_flash_attn_at_decoder=False)
    parser.add_argument('--drop-path-rate', default=0., type=float)
    parser.add_argument('--resume', default='', type=str, help='path to resume from')
    parser.add_argument('--normalize-target', action='store_true', dest='normalize_target')
    parser.add_argument('--no-normalize-target', action='store_false', dest='normalize_target')
    parser.set_defaults(normalize_target=True)
    # train
    parser.add_argument('--use-zero', action='store_true', dest='use_zero', help='use ZeRO optimizer')
    parser.add_argument('--no-use-zero', action='store_false', dest='use_zero', help='use ZeRO optimizer')
    parser.set_defaults(use_zero=False)
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--warmup-epochs', default=40, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--batch-size', default=64, type=int, help='number of samples per-device/per-gpu')
    parser.add_argument('--optimizer', default='adamw', choices=['adamw', 'lion'], type=str)
    parser.add_argument('--lr', default=1.5e-4, type=float)
    parser.add_argument('--fix-lr', action='store_true', help='disable cosine lr decay if set True')
    parser.add_argument('--lr-start', default=1e-6, type=float, help='initial warmup lr')
    parser.add_argument('--lr-end', default=1e-5, type=float, help='minimum final lr')
    parser.add_argument('--update-freq', default=1, type=int, help='optimizer update frequency (i.e. gradient accumulation steps)')
    parser.add_argument('--wd', default=0.05, type=float)
    parser.add_argument('--betas', default=(0.9, 0.95), nargs=2, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--save-freq', default=20, type=int)
    parser.add_argument('--disable-amp', action='store_true', help='disable mixed-precision training (requires more memory and compute)')
    parser.add_argument('--grad-clip-norm', default=None, type=float)
    parser.add_argument('--use-multi-epochs-loader', action='store_true')
    # system
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--evaluate', action='store_true', help='eval only')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    return parser


def main(args):
    dist_utils.init_distributed_mode(args)
    dist_utils.random_seed(args.seed, dist_utils.get_rank())

    print("=> creating model: {}".format(args.model))
    model = getattr(model_videomae, args.model)(
        pretrained=False,
        drop_path_rate=args.drop_path_rate,
        decoder_depth=args.decoder_depth,
        use_flash_attn_at_encoder=args.use_flash_attn_at_encoder,
        use_flash_attn_at_decoder=args.use_flash_attn_at_decoder,
        use_checkpoint=args.use_grad_checkpointing,
        channel_last=args.channel_last,
    )
    model.cuda(args.gpu)

    patch_size = model.encoder.patch_embed.patch_size
    args.window_size = (args.clip_length // 2, args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], bucket_cap_mb=200)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.MSELoss().cuda(args.gpu)

    n_wd, n_non_wd = [], []
    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        if (p.ndim < 2 or 'bias' in n or
            'ln' in n or 'bn' in n or
            'pos_embed' in n or 'positional_embedding' in n
        ):
            n_non_wd.append(n)
            p_non_wd.append(p)
        else:
            n_wd.append(n)
            p_wd.append(p)

    print('parameters without wd:', n_non_wd)
    print('parameters with wd:', n_wd)
    optim_params = [{"params": p_wd, "weight_decay": args.wd},
                    {"params": p_non_wd, "weight_decay": 0}]

    total_batch_size = args.batch_size * dist_utils.get_world_size()
    args.lr = args.lr * total_batch_size / 256
    args.lr_start = args.lr_start * total_batch_size / 256
    args.lr_end = args.lr_end * total_batch_size / 256
    if args.optimizer == 'adamw':
        opt_fn = torch.optim.AdamW
    elif args.optimizer == 'lion':
        opt_fn = Lion
    else:
        raise ValueError
    if args.use_zero:
        print('Training with ZeroRedundancyOptimizer')
        optimizer = ZeroRedundancyOptimizer(
            optim_params, optimizer_class=opt_fn,
            lr=args.lr, betas=args.betas, eps=args.eps, weight_decay=args.wd
        )
    else:
        optimizer = opt_fn(optim_params, lr=args.lr, betas=args.betas,
                           eps=args.eps, weight_decay=args.wd)
    scaler = amp.GradScaler(enabled=not args.disable_amp)

    # optionally resume from a checkpoint (takes precedence over autoresume)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading resume checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
            args.start_epoch = epoch
            result = model.load_state_dict(checkpoint['state_dict'], strict=False)
            print(result)
            optimizer.load_state_dict(checkpoint['optimizer']) if 'optimizer' in checkpoint else ()
            scaler.load_state_dict(checkpoint['scaler']) if 'scaler' in checkpoint else ()
            best_acc1 = checkpoint['best_acc1'] if 'best_acc1' in checkpoint else 0.
            print("=> loaded resume checkpoint '{}' (epoch {})"
                  .format(args.resume, epoch))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        # auto-resume from latest checkpoint in output directory
        latest = os.path.join(args.output_dir, 'checkpoint.pt')
        if os.path.isfile(latest):
            print("=> loading latest checkpoint '{}'".format(latest))
            latest_checkpoint = torch.load(latest, map_location='cpu')
            args.start_epoch = latest_checkpoint['epoch']
            model.load_state_dict(latest_checkpoint['state_dict'])
            optimizer.load_state_dict(latest_checkpoint['optimizer'])
            scaler.load_state_dict(latest_checkpoint['scaler'])
            best_acc1 = latest_checkpoint['best_acc1'] if 'best_acc1' in latest_checkpoint else 0.
            print("=> loaded latest checkpoint '{}' (epoch {})"
                  .format(latest, latest_checkpoint['epoch']))

    torch.backends.cudnn.benchmark = True

    # data loading
    mean, std = [0.485 * 255, 0.456 * 255, 0.406 * 255], [0.229 * 255, 0.224 * 255, 0.225 * 255]
    normalize = K.enhance.Normalize(mean=mean, std=std)

    if args.fused_decode_crop:
        train_transform = None
    else:
        train_transform_ls = [
            Permute([3, 0, 1, 2]),
            GroupMultiScaleCrop(224, [1, .875, .75, .66]),
            torchvision.transforms.RandomHorizontalFlip(0.5),
        ]
        train_transform = torchvision.transforms.Compose(train_transform_ls)
    train_dataset = KineticsDataset(
        args.root, args.train_metadata, transform=train_transform, is_training=True, 
        clip_length=args.clip_length, clip_stride=args.clip_stride,
        threads=args.decode_threads,
        fast_rrc=False, rrc_params=(224, (0.5, 1.0)),
        fast_msc=args.fused_decode_crop, msc_params=(224, ),
        fast_cc=False, cc_params=(224, ),
        hflip_prob=0.5, vflip_prob=0.,
        mask_type='later',  # do masking in batches
        window_size=args.window_size, mask_ratio=args.mask_ratio,
        verbose=args.verbose,
    )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    if args.use_multi_epochs_loader:
        train_loader = MultiEpochsDataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=args.use_pin_memory, sampler=train_sampler, drop_last=True
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=args.use_pin_memory, sampler=train_sampler, drop_last=True
        )
    print('len(train_loader) = {}'.format(len(train_loader)))

    if args.fix_lr:
        lr_schedule = None
    else:
        lr_schedule = cosine_scheduler(
            args.lr, args.lr_end, args.epochs, len(train_loader) // args.update_freq,
            warmup_epochs=args.warmup_epochs, start_warmup_value=args.lr_start
        )

    print(args)

    print("=> beginning training")
    best_acc1 = 0.
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        train_stats = train(train_loader, normalize, model, criterion, optimizer, scaler, epoch, lr_schedule, args)

        if (epoch + 1) % args.save_freq == 0:
            print("=> saving checkpoint")
            if args.use_zero:
                print('consolidated on rank {} because of ZeRO'.format(args.rank))
                optimizer.consolidate_state_dict(0)
            dist_utils.save_on_master_v2({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict() if dist_utils.is_main_process() else None,
                    'scaler': scaler.state_dict(),
                    'args': args,
                }, epoch + 1, args.output_dir)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}

        if dist_utils.is_main_process():
            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')
        
def train(train_loader, normalize, model, criterion, optimizer, scaler, epoch, lr_schedule, args):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    model_time = AverageMeter('Model', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    metric_names = ['loss']
    iters_per_epoch = len(train_loader) // args.update_freq
    metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in metric_names])
    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, model_time, mem, *metrics.values()],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()

    for data_iter, inputs in enumerate(train_loader):
        optim_iter = data_iter // args.update_freq

        # measure data loading time
        if args.verbose:
            print('Time to train: {}'.format(datetime.datetime.now()))
        data_time.update(time.time() - end)

        # update weight decay and learning rate according to their schedule
        it = iters_per_epoch * epoch + optim_iter  # global training iteration
        for k, param_group in enumerate(optimizer.param_groups):
            if lr_schedule is not None:
                param_group['lr'] = lr_schedule[it]

        videos = inputs.cuda(args.gpu, non_blocking=True)
        if args.fused_decode_crop:
            videos = videos.permute(0, 4, 1, 2, 3)
        bool_masked_pos = TubeMaskingGeneratorGPU(videos.shape[0], args.window_size, args.mask_ratio, device=args.gpu)().flatten(1).to(torch.bool)


        if args.normalize_target:
            videos_squeeze = rearrange(videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=args.patch_size[0], p2=args.patch_size[1])
            videos_norm = (videos_squeeze - videos_squeeze.mean(dim=-2, keepdim=True)
                ) / (videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
            videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)')
        else:
            videos_patch = rearrange(videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=2, p1=args.patch_size[0], p2=args.patch_size[1])

        B, _, C = videos_patch.shape
        targets = videos_patch[bool_masked_pos].reshape(B, -1, C)

        videos = normalize(videos)

        optimizer.zero_grad()

        tic = time.time()
        # compute output
        with amp.autocast(enabled=not args.disable_amp):
            outputs = model(videos, bool_masked_pos)
            loss = criterion(outputs, target=targets)
            loss /= args.update_freq

        check_loss_nan(loss)
        scaler.scale(loss).backward()

        if (data_iter + 1) % args.update_freq != 0:
            continue

        # compute gradient and do SGD step
        if args.grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
        scaler.step(optimizer)
        scaler.update()
        model.zero_grad(set_to_none=True)

        # torch.cuda.empty_cache()
        model_time.update(time.time() - tic)

        metrics['loss'].update(loss.item(), args.batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        mem.update(torch.cuda.max_memory_allocated() // 1e9)

        if optim_iter % args.print_freq == 0:
            if args.verbose:
                print('Time to print: {}'.format(datetime.datetime.now()))
            progress.display(optim_iter)
    progress.synchronize()
    return {**{k: v.avg for k, v in metrics.items()},
            'lr': optimizer.param_groups[0]['lr']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('AVION training and evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
