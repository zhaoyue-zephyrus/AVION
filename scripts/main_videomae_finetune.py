import argparse
from collections import OrderedDict
import datetime
from functools import partial
import json
import os
import time

from einops import rearrange
import kornia as K
import numpy as np
from sklearn.metrics import top_k_accuracy_score
import torch
import torch.cuda.amp as amp
from torch.distributed.optim import ZeroRedundancyOptimizer
from timm.data.loader import MultiEpochsDataLoader
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEmaV2, accuracy, get_state_dict

from avion.data.classification_dataset import VideoClsDataset, multiple_samples_collate
import avion.models.model_videomae as model_videomae
from avion.optim.layer_decay import LayerDecayValueAssigner
from avion.optim.lion import Lion
from avion.optim.schedulers import cosine_scheduler
import avion.utils.distributed as dist_utils
from avion.utils.meters import AverageMeter, ProgressMeter
from avion.utils.misc import check_loss_nan, interpolate_pos_embed


def get_args_parser():
    parser = argparse.ArgumentParser(description='VideoMAE fine-tune', add_help=False)
    parser.add_argument('--root',
                        default='datasets/Kinetics/train_320px/',
                        type=str, help='path to train dataset root')
    parser.add_argument('--root-val',
                        default='datasets/Kinetics/val_320px/',
                        type=str, help='path to val dataset root')
    parser.add_argument('--train-metadata',
                        default='datasets/Kinetics/annotations/k400_320p_train_list.txt',
                        type=str, help='metadata for train split')
    parser.add_argument('--val-metadata',
                        default='datasets/Kinetics/annotations/k400_val_list.txt',
                        type=str, help='metadata for val split')
    parser.add_argument('--output-dir', default='./', type=str, help='output dir')
    parser.add_argument('--input-size', default=224, type=int, help='input frame size')
    parser.add_argument('--clip-length', default=16, type=int, help='clip length')
    parser.add_argument('--clip-stride', default=4, type=int, help='clip stride')
    parser.add_argument('--use-pin-memory', action='store_true', dest='use_pin_memory')
    parser.add_argument('--disable-pin-memory', action='store_false', dest='use_pin_memory')
    parser.set_defaults(use_pin_memory=False)
    parser.add_argument('--nb-classes', default=400, type=int)
    # augmentation
    parser.add_argument('--repeated-aug', default=1, type=int)
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    # mixup
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')    
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    # model
    parser.add_argument('--model', default='vit_base_patch16_224', type=str)
    parser.add_argument('--channel-last', action='store_true', dest='channel_last')
    parser.add_argument('--disable-channel-last', action='store_false', dest='channel_last')
    parser.set_defaults(channel_last=False)
    parser.add_argument('--grad-checkpointing', action='store_true', dest='use_grad_checkpointing')
    parser.add_argument('--no-grad-checkpointing', action='store_false', dest='use_grad_checkpointing')
    parser.set_defaults(use_grad_checkpointing=False)
    parser.add_argument('--use-flash-attn', action='store_true', dest='use_flash_attn')
    parser.add_argument('--disable-flash-attn', action='store_false', dest='use_flash_attn')
    parser.set_defaults(use_flash_attn=False)
    parser.add_argument('--fc-drop-rate', default=0.0, type=float)
    parser.add_argument('--drop-rate', default=0.0, type=float)
    parser.add_argument('--attn-drop-rate', default=0.0, type=float)
    parser.add_argument('--drop-path-rate', default=0.1, type=float)
    parser.add_argument('--resume', default='', type=str, help='path to resume from')
    # fine-tune
    parser.add_argument('--finetune', default='', help='fine-tune path')
    parser.add_argument('--model-key', default='model|module|state_dict', type=str)
    # model ema
    parser.add_argument('--model-ema', action='store_true', default=False)
    parser.add_argument('--model-ema-decay', type=float, default=0.9999, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')
    # train
    parser.add_argument('--use-zero', action='store_true', dest='use_zero', help='use ZeRO optimizer')
    parser.add_argument('--no-use-zero', action='store_false', dest='use_zero', help='use ZeRO optimizer')
    parser.set_defaults(use_zero=False)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--warmup-epochs', default=5, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--batch-size', default=32, type=int, help='number of samples per-device/per-gpu')
    parser.add_argument('--optimizer', default='adamw', choices=['adamw', 'lion'], type=str)
    parser.add_argument('--lr', default=1.5e-3, type=float)
    parser.add_argument('--layer-decay', type=float, default=0.75)
    parser.add_argument('--lr-start', default=1e-6, type=float, help='initial warmup lr')
    parser.add_argument('--lr-end', default=1e-6, type=float, help='minimum final lr')
    parser.add_argument('--update-freq', default=1, type=int, help='optimizer update frequency (i.e. gradient accumulation steps)')
    parser.add_argument('--wd', default=0.05, type=float)
    parser.add_argument('--wd-end', type=float, default=None)
    parser.add_argument('--betas', default=(0.9, 0.999), nargs=2, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--save-freq', default=1, type=int)
    parser.add_argument('--val-freq', default=1, type=int)
    parser.add_argument('--disable-amp', action='store_true', help='disable mixed-precision training (requires more memory and compute)')
    parser.add_argument('--grad-clip-norm', default=None, type=float)
    parser.add_argument('--use-multi-epochs-loader', action='store_true')
    parser.add_argument('--decode-threads', default=1, type=int)
    # system
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--evaluate', action='store_true', help='eval only')
    parser.add_argument('--evaluate-batch-size', default=1, type=int, help='batch size at evaluation')
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
        num_classes=args.nb_classes,
        fc_drop_rate = args.fc_drop_rate,
        drop_rate = args.drop_rate,
        drop_path_rate=args.drop_path_rate,
        attn_drop_rate=args.attn_drop_rate,
        use_flash_attn=args.use_flash_attn,
        use_checkpoint=args.use_grad_checkpointing,
        channel_last=args.channel_last,
    )
    model.cuda(args.gpu)

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')
        print("=> Load checkpoint from %s" % args.finetune)
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                if list(checkpoint_model.keys())[0].startswith('module.'):
                    renamed_ckpt = {k[7:]: v for k, v in checkpoint_model.items()}
                    checkpoint_model = renamed_ckpt
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        for key in ['head.weight', 'head.bias']:
            if key in checkpoint_model and checkpoint_model[key].shape != model.state_dict()[key].shape:
                print("Removing key %s from pretrained checkpoint" % key)
                checkpoint_model.pop(key)

        new_dict = OrderedDict()
        for key in checkpoint_model.keys():
            if key.startswith('backbone.'):
                new_dict[key[9:]] = checkpoint_model[key]
            elif key.startswith('encoder.'):
                if args.use_flash_attn and 'attn.qkv' in key:
                    new_dict[key[8:].replace('attn.qkv', 'attn.Wqkv')] = checkpoint_model[key]
                elif args.use_flash_attn and 'attn.q_bias' in key:
                    q_bias = checkpoint_model[key]
                    v_bias = checkpoint_model[key.replace('attn.q_bias', 'attn.v_bias')]
                    new_dict[key[8:].replace('attn.q_bias', 'attn.Wqkv.bias')] = torch.cat(
                        (q_bias, torch.zeros_like(v_bias), v_bias))
                elif args.use_flash_attn and 'attn.v_bias' in key:
                    continue
                elif args.use_flash_attn and 'attn.proj' in key:
                    new_dict[key[8:].replace('attn.proj', 'attn.out_proj')] = checkpoint_model[key]
                else:
                    new_dict[key[8:]] = checkpoint_model[key]
            else:
                if args.use_flash_attn and 'attn.qkv' in key:
                    new_dict[key.replace('attn.qkv', 'attn.Wqkv')] = checkpoint_model[key]
                elif args.use_flash_attn and 'attn.q_bias' in key:
                    q_bias = checkpoint_model[key]
                    v_bias = checkpoint_model[key.replace('attn.q_bias', 'attn.v_bias')]
                    new_dict[key.replace('attn.q_bias', 'attn.Wqkv.bias')] = torch.cat(
                        (q_bias, torch.zeros_like(v_bias), v_bias))
                elif args.use_flash_attn and 'attn.v_bias' in key:
                    continue
                elif args.use_flash_attn and 'attn.proj' in key:
                    new_dict[key.replace('attn.proj', 'attn.out_proj')] = checkpoint_model[key]
                else:
                    new_dict[key] = checkpoint_model[key]
        checkpoint_model = new_dict

        if 'pos_embed' in checkpoint_model:
            new_pos_embed = interpolate_pos_embed(checkpoint_model['pos_embed'], model, args.clip_length)
            checkpoint_model['pos_embed'] = new_pos_embed
    
        result = model.load_state_dict(checkpoint_model, strict=False)
        print(result)

    model_ema = None
    if args.model_ema:
        model_ema = ModelEmaV2(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume=''
        )
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    if args.layer_decay < 1.0:
        num_layers = model.get_num_layers()
        ld_assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        ld_assigner = None

    # define loss function (criterion) and optimizer
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    skip_list = {}
    if hasattr(model, "no_weight_decay"):
        skip_list = model.no_weight_decay()
    parameter_group_names = {}
    parameter_group_vars = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        if p.ndim < 2 or 'bias' in n or n in skip_list:
            group_name = 'no_decay'
            this_wd = 0.
        else:
            group_name = 'with_decay'
            this_wd = args.wd

        if ld_assigner is not None:
            layer_id = ld_assigner.get_layer_id(n)
            group_name = 'layer_%d_%s' % (layer_id, group_name)

        if group_name not in parameter_group_names:
            if ld_assigner is not None:
                scale = ld_assigner.get_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {"weight_decay": this_wd, "params": [], "lr_scale": scale}
            parameter_group_vars[group_name] = {"weight_decay": this_wd, "params": [], "lr_scale": scale}

        parameter_group_names[group_name]["params"].append(n)
        parameter_group_vars[group_name]["params"].append(p)

    print("Param groups:", parameter_group_names)
    optim_params = parameter_group_vars.values()

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

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], bucket_cap_mb=200)

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
            best_acc1 = checkpoint['best_acc1'] if hasattr(checkpoint, 'best_acc1') else 0
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
            best_acc1 = checkpoint['best_acc1'] if hasattr(checkpoint, 'best_acc1') else 0
            print("=> loaded latest checkpoint '{}' (epoch {})"
                  .format(latest, latest_checkpoint['epoch']))

    torch.backends.cudnn.benchmark = True

    train_dataset = VideoClsDataset(
        args.root, args.train_metadata, mode='train', 
        clip_length=args.clip_length, clip_stride=args.clip_stride,
        args=args,
    )

    val_dataset = VideoClsDataset(
        args.root_val, args.val_metadata, mode='validation',
        clip_length=args.clip_length, clip_stride=args.clip_stride,
        args=args,
    )

    test_dataset = VideoClsDataset(
        args.root_val, args.val_metadata, mode='test',
        clip_length=args.clip_length, clip_stride=args.clip_stride,
        test_num_segment=5, test_num_crop=3,
        args=args,
    )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler, val_sampler, test_sampler = None, None, None

    if args.repeated_aug > 1:
        collate_func = partial(multiple_samples_collate, fold=False)
    else:
        collate_func = None

    if args.use_multi_epochs_loader:
        train_loader = MultiEpochsDataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=args.use_pin_memory, sampler=train_sampler, drop_last=True,
            collate_fn=collate_func,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=args.use_pin_memory, sampler=train_sampler, drop_last=True,
            collate_fn=collate_func,
        )
    print('len(train_loader) = {}'.format(len(train_loader)))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False, sampler=val_sampler, drop_last=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.evaluate_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False, sampler=test_sampler, drop_last=False
    )


    lr_schedule = cosine_scheduler(
        args.lr, args.lr_end, args.epochs, len(train_loader) // args.update_freq,
        warmup_epochs=args.warmup_epochs, start_warmup_value=args.lr_start
    )
    if args.wd_end is None:
        args.wd_end = args.wd
    wd_schedule = cosine_scheduler(args.wd, args.wd_end, args.epochs, len(train_loader) // args.update_freq)


    print(args)

    if args.evaluate:
        _ = test(test_loader, model, args, len(test_dataset))
        return

    print("=> beginning training")
    best_acc1 = 0.
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        train_stats = train(
            train_loader, model, criterion, optimizer, 
            scaler, epoch, model_ema, mixup_fn,
            lr_schedule, wd_schedule, args,
        )

        if (epoch + 1) % args.save_freq == 0:
            print("=> saving checkpoint")
            if args.use_zero:
                print('consolidated on rank {} because of ZeRO'.format(args.rank))
                optimizer.consolidate_state_dict(to=args.rank)
            dist_utils.save_on_master({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'args': args,
                    'model_ema': get_state_dict(model_ema) if model_ema is not None else None,
                }, False, args.output_dir)
        if (epoch + 1) % args.val_freq == 0:
            print("=> validate")
            val_stats = validate(val_loader, model, epoch, args)
            if best_acc1 < val_stats['acc1']:
                best_acc1 = val_stats['acc1']
                if args.use_zero:
                    print('consolidated on rank {} because of ZeRO'.format(args.rank))
                    optimizer.consolidate_state_dict(to=args.rank)
                dist_utils.save_on_master({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scaler': scaler.state_dict(),
                        'best_acc1': best_acc1,
                        'args': args,
                        'model_ema': get_state_dict(model_ema) if model_ema is not None else None,
                    }, True, args.output_dir)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}

        if dist_utils.is_main_process():
            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')
        
def train(train_loader, model, criterion, optimizer,
          scaler, epoch, model_ema, mixup_fn,
          lr_schedule, wd_schedule, args):
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
                param_group['lr'] = lr_schedule[it] * param_group["lr_scale"]
            if wd_schedule is not None and param_group['weight_decay'] > 0:
                param_group['weight_decay'] = wd_schedule[it]

        videos = inputs[0].cuda(args.gpu, non_blocking=True)
        targets = inputs[1].cuda(args.gpu, non_blocking=True)

        if mixup_fn is not None:
            videos, targets = mixup_fn(videos, targets)

        optimizer.zero_grad()

        tic = time.time()
        # compute output
        with amp.autocast(enabled=not args.disable_amp):
            outputs = model(videos)
            loss = criterion(outputs, targets)
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
        if model_ema is not None:
            model_ema.update(model)

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


def validate(val_loader, model, epoch, args):
    criterion = torch.nn.CrossEntropyLoss()
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    model_time = AverageMeter('Model', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    metric_names = ['loss', 'acc1', 'acc5']
    metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in metric_names])
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time, model_time, mem, *metrics.values()],
        prefix="Epoch: [{}]".format(epoch))

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for data_iter, inputs in enumerate(val_loader):
            data_time.update(time.time() - end)

            videos = inputs[0].cuda(args.gpu, non_blocking=True)
            targets = inputs[1].cuda(args.gpu, non_blocking=True)

            tic = time.time()
            # compute output
            with amp.autocast(enabled=not args.disable_amp):
                outputs = model(videos)
                loss = criterion(outputs, targets)

            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

            # torch.cuda.empty_cache()
            model_time.update(time.time() - tic)

            metrics['loss'].update(loss.item(), args.batch_size)
            metrics['acc1'].update(acc1.item(), args.batch_size)
            metrics['acc5'].update(acc5.item(), args.batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % args.print_freq == 0:
                progress.display(data_iter)
    progress.synchronize()
    return {k: v.avg for k, v in metrics.items()}


def test(test_loader, model, args, num_videos):
    criterion = torch.nn.CrossEntropyLoss()
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    model_time = AverageMeter('Model', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    metric_names = ['loss', 'acc1', 'acc5']
    metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in metric_names])
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, data_time, model_time, mem, *metrics.values()],
        prefix="Testing: ")

    # switch to eval mode
    model.eval()

    all_logits = [[] for _ in range(args.world_size)]
    all_probs = [[] for _ in range(args.world_size)]
    all_targets = [[] for _ in range(args.world_size)]
    total_num = 0
    with torch.no_grad():
        end = time.time()
        for data_iter, inputs in enumerate(test_loader):
            data_time.update(time.time() - end)

            videos = inputs[0].cuda(args.gpu, non_blocking=True)
            targets = inputs[1].cuda(args.gpu, non_blocking=True)
            this_batch_size = videos.shape[0]

            tic = time.time()
            # compute output
            with amp.autocast(enabled=not args.disable_amp):
                targets_repeated = torch.repeat_interleave(targets, videos.shape[1])
                videos = rearrange(videos, 'b n t c h w -> (b n) t c h w')
                logits = model(videos)
                loss = criterion(logits, targets_repeated)

            acc1, acc5 = accuracy(logits, targets_repeated, topk=(1, 5))

            logits = rearrange(logits, '(b n) k -> b n k', b=this_batch_size)
            probs = torch.softmax(logits, dim=2)
            gathered_logits = [torch.zeros_like(logits) for _ in range(args.world_size)]
            gathered_probs = [torch.zeros_like(probs) for _ in range(args.world_size)]
            gathered_targets = [torch.zeros_like(targets) for _ in range(args.world_size)]
            torch.distributed.all_gather(gathered_logits, logits)
            torch.distributed.all_gather(gathered_probs, probs)
            torch.distributed.all_gather(gathered_targets, targets)
            for j in range(args.world_size):
                all_logits[j].append(gathered_logits[j].detach().cpu())
                all_probs[j].append(gathered_probs[j].detach().cpu())
                all_targets[j].append(gathered_targets[j].detach().cpu())

            # torch.cuda.empty_cache()
            model_time.update(time.time() - tic)

            metrics['loss'].update(loss.item(), this_batch_size)
            metrics['acc1'].update(acc1.item(), this_batch_size)
            metrics['acc5'].update(acc5.item(), this_batch_size)
            total_num += logits.shape[0] * args.world_size

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % args.print_freq == 0:
                progress.display(data_iter)
    progress.synchronize()
    for j in range(args.world_size):
        all_logits[j] = torch.cat(all_logits[j], dim=0).numpy()
        all_probs[j] = torch.cat(all_probs[j], dim=0).numpy()
        all_targets[j] = torch.cat(all_targets[j], dim=0).numpy()
    all_logits_reorg, all_probs_reorg, all_targets_reorg = [], [], []
    for i in range(total_num):
        all_logits_reorg.append(all_logits[i % args.world_size][i // args.world_size])
        all_probs_reorg.append(all_probs[i % args.world_size][i // args.world_size])
        all_targets_reorg.append(all_targets[i % args.world_size][i // args.world_size])
    all_logits = np.stack(all_logits_reorg, axis=0)
    all_probs = np.stack(all_probs_reorg, axis=0)
    all_targets = np.stack(all_targets_reorg, axis=0)
    all_logits = all_logits[:num_videos, :].mean(axis=1)
    all_probs = all_probs[:num_videos, :].mean(axis=1)
    all_targets = all_targets[:num_videos, ]
    acc1 = top_k_accuracy_score(all_targets, all_logits, k=1)
    acc5 = top_k_accuracy_score(all_targets, all_logits, k=5)
    print('[Average logits] Overall top1={:.3f} top5={:.3f}'.format(acc1, acc5))
    acc1 = top_k_accuracy_score(all_targets, all_probs, k=1)
    acc5 = top_k_accuracy_score(all_targets, all_probs, k=5)
    print('[Average probs ] Overall top1={:.3f} top5={:.3f}'.format(acc1, acc5))
    return {k: v.avg for k, v in metrics.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('AVION training and evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
