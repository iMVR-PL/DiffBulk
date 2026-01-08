# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Main training loop."""

import os
import time
import copy
import pickle
import psutil
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import persistence
from torch_utils import misc
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from training.dataset import GenePatchDataset
#----------------------------------------------------------------------------
# Uncertainty-based loss function (Equations 14,15,16,21) proposed in the
# paper "Analyzing and Improving the Training Dynamics of Diffusion Models".

@persistence.persistent_class
class EDM2Loss:
    def __init__(self, P_mean=-0.4, P_std=1.0, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, rand_seed=42):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()  
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        noise = torch.randn_like(images) * sigma
        denoised, logvar = net(images + noise, sigma, is_sampling=False, gene_labels=labels, return_logvar=True, rand_seed=rand_seed)
        loss = (weight / logvar.exp()) * ((denoised - images) ** 2) + logvar
        return loss  # [N, C, H, W]

#----------------------------------------------------------------------------
# Learning rate decay schedule used in the paper "Analyzing and Improving
# the Training Dynamics of Diffusion Models".

def learning_rate_schedule(cur_nimg, batch_size, ref_lr=100e-4, ref_batches=70e3, rampup_Mimg=10):
    lr = ref_lr  # 0.0100
    if ref_batches > 0:
        lr /= np.sqrt(max(cur_nimg / (ref_batches * batch_size), 1))
    if rampup_Mimg > 0:
        lr *= min(cur_nimg / (rampup_Mimg * 1e6), 1)
    return lr

#----------------------------------------------------------------------------
# Main training loop.

def training_loop(
    dataset_kwargs      = dict(class_name='training.dataset.GenePatchDataset', path=None),
    encoder_kwargs      = dict(class_name='training.encoders.StandardRGBEncoder'),  # StabilityVAEEncoder
    data_loader_kwargs  = dict(class_name='torch.utils.data.DataLoader', pin_memory=True, num_workers=2, prefetch_factor=2),
    network_kwargs      = dict(class_name='training.networks_edm2.Precond'),
    loss_kwargs         = dict(class_name='training.training_loop.EDM2Loss'),
    optimizer_kwargs    = dict(class_name='torch.optim.Adam', betas=(0.9, 0.99)),
    lr_kwargs           = dict(func_name='training.training_loop.learning_rate_schedule'),
    ema_kwargs          = dict(class_name='training.phema.PowerFunctionEMA'),

    run_dir             = '.',      # Output directory.
    seed                = 0,        # Global random seed.
    batch_size          = 2048,     # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU. None = no limit.
    total_nimg          = 8<<30,    # Train for a total of N training images.
    slice_nimg          = None,     # Train for a maximum of N training images in one invocation. None = no limit.
    status_nimg         = 128<<10,  # Report status every N training images. None = disable.
    snapshot_nimg       = 8<<20,    # Save network snapshot every N training images. None = disable.
    checkpoint_nimg     = 128<<20,  # Save state checkpoint every N training images. None = disable.
    log_interval_nimg   = 1<<13,

    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    force_finite        = True,     # Get rid of NaN/Inf gradients before feeding them to the optimizer.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),

    valid_dataset_kwargs = dict(class_name='training.dataset.GenePatchDataset', path=None),  # valid dataset
    valid_interval_nimg = 128<<10,
    valid_batch_size = 256,
):
    # Initialize.
    prev_status_time = time.time()

    # Initialize SummaryWriter only for rank 0
    if dist.get_rank() == 0:
        writer = SummaryWriter(log_dir=run_dir)

    misc.set_random_seed(seed, dist.get_rank())
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Validate batch size.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()
    assert total_nimg % batch_size == 0
    assert slice_nimg is None or slice_nimg % batch_size == 0
    assert status_nimg is None or status_nimg % batch_size == 0
    assert snapshot_nimg is None or (snapshot_nimg % batch_size == 0 and snapshot_nimg % 1024 == 0)
    assert checkpoint_nimg is None or (checkpoint_nimg % batch_size == 0 and checkpoint_nimg % 1024 == 0)
  
    # Load previous checkpoint and decide how long to train.
    dist.print0('Setting up training state...')
    state = dnnlib.EasyDict(cur_nimg=0, total_elapsed_time=0)
    
    # Setup dataset, encoder, and network.
    dist.print0('Loading dataset...')
    datasets = []
    dataset_iterators = []
    
    for patch_path, gene_path in zip(dataset_kwargs.patch_path, dataset_kwargs.gene_path):
        # Create dataset
        dataset = GenePatchDataset(patch_path=patch_path, gene_path=gene_path)
        datasets.append(dataset)
        
        # Create sampler
        sampler = misc.InfiniteSampler(
            dataset=dataset, 
            rank=dist.get_rank(), 
            num_replicas=dist.get_world_size(), 
            seed=seed, 
            start_idx=state.cur_nimg
        )

        # Create iterator
        dataset_iterator = iter(dnnlib.util.construct_class_by_name(
            dataset=dataset, 
            sampler=sampler, 
            batch_size=batch_gpu, 
            **data_loader_kwargs
        ))
        dataset_iterators.append(dataset_iterator)

    # Access each dataset, sampler, and iterator from the lists
    dataset_iterator_1, dataset_iterator_2, dataset_iterator_3 = dataset_iterators

    ref_image, ref_label = datasets[0][0]  # tensor 255
    dist.print0('Setting up encoder...')
    encoder = dnnlib.util.construct_class_by_name(**encoder_kwargs)
    ref_image = encoder.encode_latents(torch.as_tensor(ref_image).to(device).unsqueeze(0))
    dist.print0(f'Image shape: {ref_image.shape}...')  # tensor [1, 3, 224, 224]
    dist.print0(f'label shape: {ref_label.shape}...')  # tensor []
    dist.print0('Constructing network...')
    interface_kwargs = dict(img_resolution=ref_image.shape[-1], img_channels=ref_image.shape[1])
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs)
    net.train().requires_grad_(True).to(device)
    
    # Print network summary.
    if dist.get_rank() == 0:
        misc.print_module_summary(net, [
            torch.zeros([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device),
            torch.ones([batch_gpu], device=device),
            torch.zeros([batch_gpu, 460], device=device),
            False,
        ], max_nesting=2)
    
    # initial validation dataset
    dist.print0('Loading validation dataset...')

    # Initialize lists for datasets, samplers, and dataloaders
    valid_data_loaders = []

    # Loop over the patch and gene paths to create datasets, samplers, and dataloaders
    for patch_path, gene_path in zip(valid_dataset_kwargs.patch_path, valid_dataset_kwargs.gene_path):
        # Create dataset
        valid_dataset = GenePatchDataset(patch_path=patch_path, gene_path=gene_path)

        # Create sampler
        valid_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset=valid_dataset, 
            shuffle=False,
            rank=dist.get_rank(),
            num_replicas=dist.get_world_size()
        )

        # Create dataloader
        valid_data_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=valid_batch_size,
            sampler=valid_sampler,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2
        )
        valid_data_loaders.append(valid_data_loader)

    # Dataloader from the lists
    valid_data_loader_1, valid_data_loader_2, valid_data_loader_3 = valid_data_loaders

    # Setup training state.
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], find_unused_parameters=True)
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs)
    ema = dnnlib.util.construct_class_by_name(net=net, **ema_kwargs) if ema_kwargs is not None else None

    # Load previous checkpoint and decide how long to train.
    checkpoint = dist.CheckpointIO(state=state, net=net, loss_fn=loss_fn, optimizer=optimizer, ema=ema)
    checkpoint.load_latest(run_dir)
    stop_at_nimg = total_nimg
    if slice_nimg is not None:  # None
        granularity = checkpoint_nimg if checkpoint_nimg is not None else snapshot_nimg if snapshot_nimg is not None else batch_size
        slice_end_nimg = (state.cur_nimg + slice_nimg) // granularity * granularity # round down
        stop_at_nimg = min(stop_at_nimg, slice_end_nimg)
    assert stop_at_nimg > state.cur_nimg
    dist.print0(f'Training from {state.cur_nimg // 1000} kimg to {stop_at_nimg // 1000} kimg:')
    dist.print0()

    
    def validate():
        """Run validation and log metrics."""
        dist.print0("Running validation...")
        net.eval()
        # Ensure consistent data splitting across processes
        valid_data_loader_1.sampler.set_epoch(0)
        valid_data_loader_2.sampler.set_epoch(0)
        valid_data_loader_3.sampler.set_epoch(0)

        valid_loss_total_1 = torch.tensor(0.0, device=device)
        valid_loss_total_2 = torch.tensor(0.0, device=device)
        valid_loss_total_3 = torch.tensor(0.0, device=device)

        valid_images_count_1 = torch.tensor(0, device=device)
        valid_images_count_2 = torch.tensor(0, device=device)
        valid_images_count_3 = torch.tensor(0, device=device)

        with torch.no_grad():
            for valid_patches, valid_genes in valid_data_loader_1:
                valid_patches = encoder.encode_latents(valid_patches.to(device))
                loss = loss_fn(net=ddp, images=valid_patches, labels=valid_genes.to(device), rand_seed=seed)
                valid_loss_total_1 += loss.sum()
                valid_images_count_1 += valid_patches.size(0)

            for valid_patches, valid_genes in valid_data_loader_2:
                valid_patches = encoder.encode_latents(valid_patches.to(device))
                loss = loss_fn(net=ddp, images=valid_patches, labels=valid_genes.to(device), rand_seed=seed)
                valid_loss_total_2 += loss.sum()
                valid_images_count_2 += valid_patches.size(0)

            for valid_patches, valid_genes in valid_data_loader_3:
                valid_patches = encoder.encode_latents(valid_patches.to(device))
                loss = loss_fn(net=ddp, images=valid_patches, labels=valid_genes.to(device), rand_seed=seed)
                valid_loss_total_3 += loss.sum()
                valid_images_count_3 += valid_patches.size(0)

        # reduce metrics from GPUs
        total_metrics_1 = torch.stack([valid_loss_total_1, valid_images_count_1])
        total_metrics_2 = torch.stack([valid_loss_total_2, valid_images_count_2])
        total_metrics_3 = torch.stack([valid_loss_total_3, valid_images_count_3])
        torch.distributed.all_reduce(total_metrics_1, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_metrics_2, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_metrics_3, op=torch.distributed.ReduceOp.SUM)

        # Compute validation loss
        valid_loss_total_1, valid_images_count_1 = total_metrics_1[0], total_metrics_1[1]
        valid_loss_total_2, valid_images_count_2 = total_metrics_2[0], total_metrics_2[1]
        valid_loss_total_3, valid_images_count_3 = total_metrics_3[0], total_metrics_3[1]

        avg_valid_loss_1 = valid_loss_total_1.item() / valid_images_count_1.item()
        avg_valid_loss_2 = valid_loss_total_2.item() / valid_images_count_2.item()
        avg_valid_loss_3 = valid_loss_total_3.item() / valid_images_count_3.item()

        dist.print0(f"Validation Loss 1: {avg_valid_loss_1}")
        dist.print0(f"Validation Loss 2: {avg_valid_loss_2}")
        dist.print0(f"Validation Loss 3: {avg_valid_loss_3}")

        # Logging validation loss
        if dist.get_rank() == 0:
            writer.add_scalar('Loss/valid_loss_1', avg_valid_loss_1, state.cur_nimg // 1000)
            writer.add_scalar('Loss/valid_loss_2', avg_valid_loss_2, state.cur_nimg // 1000)
            writer.add_scalar('Loss/valid_loss_3', avg_valid_loss_3, state.cur_nimg // 1000)

        net.train()

    # Main training loop.
    # Initialize epoch tracking variables
    prev_status_nimg = state.cur_nimg
    cumulative_training_time = 0
    start_nimg = state.cur_nimg
    prev_log_nimg = state.cur_nimg
    stats_jsonl = None

    while True:
        done = (state.cur_nimg >= stop_at_nimg)

        # Generate a unique seed for this batch and pass it to the loss function and network
        rand_seed = seed + state.cur_nimg

        # Report status.
        if status_nimg is not None and (done or state.cur_nimg % status_nimg == 0) and (state.cur_nimg != start_nimg or start_nimg == 0):
            cur_time = time.time()
            state.total_elapsed_time += cur_time - prev_status_time
            cur_process = psutil.Process(os.getpid())
            cpu_memory_usage = sum(p.memory_info().rss for p in [cur_process] + cur_process.children(recursive=True))
            dist.print0(' '.join(['Status:',
                'kimg',         f"{training_stats.report0('Progress/kimg',                              state.cur_nimg / 1e3):<9.1f}",
                'time',         f"{dnnlib.util.format_time(training_stats.report0('Timing/total_sec',   state.total_elapsed_time)):<12s}",
                'sec/tick',     f"{training_stats.report0('Timing/sec_per_tick',                        cur_time - prev_status_time):<8.2f}",
                'sec/kimg',     f"{training_stats.report0('Timing/sec_per_kimg',                        cumulative_training_time / max(state.cur_nimg - prev_status_nimg, 1) * 1e3):<7.3f}",
                'maintenance',  f"{training_stats.report0('Timing/maintenance_sec',                     cur_time - prev_status_time - cumulative_training_time):<7.2f}",
                'cpumem',       f"{training_stats.report0('Resources/cpu_mem_gb',                       cpu_memory_usage / 2**30):<6.2f}",
                'gpumem',       f"{training_stats.report0('Resources/peak_gpu_mem_gb',                  torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}",
                'reserved',     f"{training_stats.report0('Resources/peak_gpu_mem_reserved_gb',         torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}",
            ]))
            cumulative_training_time = 0
            prev_status_nimg = state.cur_nimg
            prev_status_time = cur_time
            torch.cuda.reset_peak_memory_stats()

            # Flush training stats.
            training_stats.default_collector.update()
            if dist.get_rank() == 0:
                if stats_jsonl is None:
                    stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
                fmt = {'Progress/tick': '%.0f', 'Progress/kimg': '%.3f', 'timestamp': '%.3f'}
                items = [(name, value.mean) for name, value in training_stats.default_collector.as_dict().items()] + [('timestamp', time.time())]
                items = [f'"{name}": ' + (fmt.get(name, '%g') % value if np.isfinite(value) else 'NaN') for name, value in items]
                stats_jsonl.write('{' + ', '.join(items) + '}\n')
                stats_jsonl.flush()

            # Update progress and check for abort.
            dist.update_progress(state.cur_nimg // 1000, stop_at_nimg // 1000)  # pass
            if state.cur_nimg == stop_at_nimg and state.cur_nimg < total_nimg:
                dist.request_suspend()  # pass
            if dist.should_stop() or dist.should_suspend():  # False
                done = True

        # Save network snapshot.
        if snapshot_nimg is not None and state.cur_nimg % snapshot_nimg == 0 and (state.cur_nimg != start_nimg or start_nimg == 0) and dist.get_rank() == 0:
            ema_list = ema.get() if ema is not None else optimizer.get_ema(net) if hasattr(optimizer, 'get_ema') else net
            ema_list = ema_list if isinstance(ema_list, list) else [(ema_list, '')]
            for ema_net, ema_suffix in ema_list:
                data = dnnlib.EasyDict(encoder=encoder, dataset_kwargs=dataset_kwargs, loss_fn=loss_fn)
                data.ema = copy.deepcopy(ema_net).cpu().eval().requires_grad_(False).to(torch.float16)
                fname = f'network-snapshot-{state.cur_nimg//1000:07d}{ema_suffix}.pkl'
                dist.print0(f'Saving {fname} ... ', end='', flush=True)
                with open(os.path.join(run_dir, fname), 'wb') as f:
                    pickle.dump(data, f)
                dist.print0('done')
                del data # conserve memory

        # Save state checkpoint.
        # if checkpoint_nimg is not None and (done or state.cur_nimg % checkpoint_nimg == 0) and state.cur_nimg != start_nimg:
        #     checkpoint.save(os.path.join(run_dir, f'training-state-{state.cur_nimg//1000:07d}.pt'))
        #     misc.check_ddp_consistency(net)

        # Done?
        if done:  # False
            break
        
        # validate
        if valid_interval_nimg is not None and state.cur_nimg % valid_interval_nimg == 0:
            validate()

        # Evaluate loss and accumulate gradients.
        dataset_iterators = [dataset_iterator_1, dataset_iterator_2, dataset_iterator_3]
        batch_start_time = time.time()
        misc.set_random_seed(seed, dist.get_rank(), state.cur_nimg)
        optimizer.zero_grad(set_to_none=True)
        loss_value_total = 0.0  # accumulate loss
        num_imgs = 0
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                iterator = dataset_iterators[(state.cur_nimg // batch_size + round_idx) % len(dataset_iterators)]
                patches, genes = next(iterator)  # tensor 0~255

                patches = encoder.encode_latents(patches.to(device))  # tensor -1.~1.
                genes = genes.to(device)

                loss = loss_fn(net=ddp, images=patches, labels=genes, rand_seed=rand_seed)
                training_stats.report('Loss/loss', loss)
                # Detach loss before summing and converting to item
                loss_value = loss.detach().sum().item()
                loss_value_total += loss_value
                loss.sum().mul(loss_scaling / batch_gpu_total).backward()
                num_imgs += patches.size(0)

        # Run optimizer and update weights.
        lr = dnnlib.util.call_func_by_name(cur_nimg=state.cur_nimg, batch_size=batch_size, **lr_kwargs)
        training_stats.report('Loss/learning_rate', lr)
        for g in optimizer.param_groups:
            g['lr'] = lr
        if force_finite:  # True
            for param in net.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)
        optimizer.step()

        # Update EMA and training state.
        state.cur_nimg += batch_size
        if ema is not None:
            ema.update(cur_nimg=state.cur_nimg, batch_size=batch_size)
        cumulative_training_time += time.time() - batch_start_time

        # log training loss
        if (state.cur_nimg - prev_log_nimg) >= log_interval_nimg:
            if dist.get_rank() == 0:  
                avg_loss = loss_value_total / num_imgs
                writer.add_scalar('Loss/train_loss', avg_loss, state.cur_nimg // 1000)
                print(f'Training Loss at {state.cur_nimg // 1000} kimg: {avg_loss}')
            prev_log_nimg = state.cur_nimg

#----------------------------------------------------------------------------
