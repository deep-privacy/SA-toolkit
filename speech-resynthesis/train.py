# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/jik876/hifi-gan

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', message='.*kernel_size exceeds volume extent.*')

import logging
logging.getLogger('matplotlib').setLevel(level=logging.CRITICAL)

import itertools
import os
import sys
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from dataset import CodeDataset, mel_spectrogram, get_dataset_filelist
from models import CodeGenerator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss, \
    discriminator_loss
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, \
    save_checkpoint, build_env, AttrDict

from torch._six import container_abcs, string_classes, int_classes

#  torch.backends.cudnn.benchmark = True

def collectFn(batch):
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        if len(elem.shape) == 2:
            lengths = torch.tensor([t.shape[1] for t in batch])
            if lengths.sum().item() != batch[0].shape[1] * len(batch):
                logging.warning("Padding tensor lengths={}".format(str(lengths)))
                l = [i.transpose(0, 1) for i in batch]
                return torch.nn.utils.rnn.pad_sequence(l, batch_first=True).transpose(1, 2)
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return collectFn([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: collectFn([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(collectFn(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [collectFn(samples) for samples in transposed]
    raise TypeError(
        "collectFn: batch must contain tensors, numpy arrays, numbers, dicts or lists; found {}".format(
            elem_type
        )
    )

def train(rank, local_rank, a, h):
    if h.num_gpus > 1:
        init_process_group(
            backend=h.dist_config['dist_backend'],
            init_method=h.dist_config['dist_url'],
            rank=rank,
            world_size=h.num_gpus,
        )

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(local_rank))

    generator = CodeGenerator(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    if rank == 0:
        print(generator)
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        steps = state_dict_do['steps'] + 1
        steps = state_dict_do['steps']
        last_epoch = state_dict_do['epoch']

    if h.num_gpus > 1:
        generator = DistributedDataParallel(
            generator,
            device_ids=[local_rank],
            find_unused_parameters=('f0_quantizer' in h),
        ).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[local_rank]).to(device)
        msd = DistributedDataParallel(msd, device_ids=[local_rank]).to(device)

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()), h.learning_rate,
                                betas=[h.adam_b1, h.adam_b2])

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist(h)

    trainset = CodeDataset(training_filelist, h.segment_size, h.code_hop_size, h.n_fft, h.num_mels, h.hop_size,
                           h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0, fmax_loss=h.fmax_for_loss,
                           device=device, f0=h.get('f0', None), multispkr=h.get('multispkr', None),
                           f0_stats=h.get('f0_stats', None),
                           f0_normalize=h.get('f0_normalize', False), f0_feats=h.get('f0_feats', False),
                           f0_median=h.get('f0_median', False), f0_interp=h.get('f0_interp', False),
                           vqvae=h.get('code_vq_params', False))

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False, sampler=train_sampler,
                              batch_size=h.batch_size, pin_memory=True, drop_last=True, collate_fn=collectFn)

    if rank == 0:
        validset = CodeDataset(validation_filelist, h.segment_size, h.code_hop_size, h.n_fft, h.num_mels, h.hop_size,
                               h.win_size, h.sampling_rate, h.fmin, h.fmax, False, n_cache_reuse=0,
                               fmax_loss=h.fmax_for_loss, device=device, f0=h.get('f0', None),
                               multispkr=h.get('multispkr', None),
                               f0_stats=h.get('f0_stats', None), f0_normalize=h.get('f0_normalize', False),
                               f0_feats=h.get('f0_feats', False), f0_median=h.get('f0_median', False),
                               f0_interp=h.get('f0_interp', False), vqvae=h.get('code_vq_params', False))
        validation_loader = DataLoader(validset, num_workers=h.num_workers, shuffle=False, sampler=None,
                                       batch_size=h.batch_size, pin_memory=True, drop_last=True, collate_fn=collectFn)

        audio_example_set = CodeDataset(validation_filelist, -1, h.code_hop_size, h.n_fft, h.num_mels, h.hop_size,
                               h.win_size, h.sampling_rate, h.fmin, h.fmax, False, n_cache_reuse=0,
                               fmax_loss=h.fmax_for_loss, device=device, f0=h.get('f0', None),
                               multispkr=h.get('multispkr', None),
                               f0_stats=h.get('f0_stats', None), f0_normalize=h.get('f0_normalize', False),
                               f0_feats=h.get('f0_feats', False), f0_median=h.get('f0_median', False),
                               f0_interp=h.get('f0_interp', False), vqvae=h.get('code_vq_params', False))

        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    generator.train()
    mpd.train()
    msd.train()

    #  Check dataloader working for all iter
    #  for i, batch in enumerate(train_loader):
        #  x, y, _, y_mel = batch


    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch + 1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            x, y, _, y_mel = batch
            y = torch.autograd.Variable(y.to(device, non_blocking=False))
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=False))
            y = y.unsqueeze(1)
            x = {k: torch.autograd.Variable(v.to(device, non_blocking=False)) for k, v in x.items()}

            y_g_hat = generator(**x)
            if h.get('f0_vq_params', None) or h.get('code_vq_params', None):
                y_g_hat, commit_losses, metrics = y_g_hat

            assert y_g_hat.shape[2] >= y.shape[2] - 10 and y_g_hat.shape[2] <= y.shape[2] + 9000, f"Mismatch too high in vocoder output shape - {y_g_hat.shape} != {y.shape}"
            y_g_hat = y_g_hat[:,:,:y.shape[2]]
            if h.get('f0_vq_params', None):
                f0_commit_loss = commit_losses[1][0]
                f0_metrics = metrics[1][0]
            if h.get('code_vq_params', None):
                code_commit_loss = commit_losses[0][0]
                code_metrics = metrics[0][0]

            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size,
                                          h.win_size, h.fmin, h.fmax_for_loss)

            optim_d.zero_grad()

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_s + loss_disc_f

            loss_disc_all.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
            if h.get('f0_vq_params', None):
                loss_gen_all += f0_commit_loss * h.get('lambda_commit', None)
            if h.get('code_vq_params', None):
                loss_gen_all += code_commit_loss * h.get('lambda_commit_code', None)

            loss_gen_all.backward()
            optim_g.step()

            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()

                    print(
                        'Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.format(steps,
                                                                                                                  loss_gen_all,
                                                                                                                  mel_error,
                                                                                                                  time.time() - start_b))

                # checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path, {'mpd': (mpd.module if h.num_gpus > 1 else mpd).state_dict(),
                                                      'msd': (msd.module if h.num_gpus > 1 else msd).state_dict(),
                                                      'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(),
                                                      'steps': steps, 'epoch': epoch})

                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                    sw.add_scalar("training/mel_spec_error", mel_error, steps)
                    if h.get('f0_vq_params', None):
                        sw.add_scalar("training/commit_error", f0_commit_loss, steps)
                        sw.add_scalar("training/used_curr", f0_metrics['used_curr'].item(), steps)
                        sw.add_scalar("training/entropy", f0_metrics['entropy'].item(), steps)
                        sw.add_scalar("training/usage", f0_metrics['usage'].item(), steps)
                    if h.get('code_vq_params', None):
                        sw.add_scalar("training/code_commit_error", code_commit_loss, steps)
                        sw.add_scalar("training/code_used_curr", code_metrics['used_curr'].item(), steps)
                        sw.add_scalar("training/code_entropy", code_metrics['entropy'].item(), steps)
                        sw.add_scalar("training/code_usage", code_metrics['usage'].item(), steps)

                # Validation
                if steps % a.validation_interval == 0:  # and steps != 0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            x, y, _, y_mel = batch
                            x = {k: v.to(device, non_blocking=False) for k, v in x.items()}

                            y_g_hat = generator(**x)
                            if h.get('f0_vq_params', None) or h.get('code_vq_params', None):
                                y_g_hat, commit_losses, _ = y_g_hat

                            if h.get('f0_vq_params', None):
                                f0_commit_loss = commit_losses[1][0]
                                val_err_tot += f0_commit_loss * h.get('lambda_commit', None)

                            if h.get('code_vq_params', None):
                                code_commit_loss = commit_losses[0][0]
                                val_err_tot += code_commit_loss * h.get('lambda_commit_code', None)
                            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=False))

                            assert y_g_hat.shape[2] >= y.shape[1] - 10 and y_g_hat.shape[2] <= y.shape[1] + 8000, f"Mismatch too high in vocoder output shape - {y_g_hat.shape} != {y.shape}"
                            y_g_hat = y_g_hat[:,:,:y.shape[1]]

                            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                                                          h.hop_size, h.win_size, h.fmin, h.fmax_for_loss)

                            val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

                            if j <= 4:
                                x, y, _, y_mel = audio_example_set[j]
                                x = {k: (v).to(device, non_blocking=False).unsqueeze(0) for k, v in x.items()}
                                y_g_hat = generator(**x)

                                if steps == 0:
                                    print("Len generated audio:", y_g_hat.shape, "Len GT:", len(y))
                                    sw.add_audio('gt/y_{}'.format(j), y, steps, h.sampling_rate)
                                    sw.add_figure('gt/y_spec_{}'.format(j), plot_spectrogram(y_mel.cpu()), steps)

                                sw.add_audio('generated/y_hat_{}'.format(j), y_g_hat, steps, h.sampling_rate)
                                y_hat_spec = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels,
                                                             h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)
                                sw.add_figure('generated/y_hat_spec_{}'.format(j),
                                              plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy()), steps)

                        val_err = val_err_tot / (j + 1)
                        sw.add_scalar("validation/mel_spec_error", val_err, steps)
                        if h.get('f0_vq_params', None):
                            sw.add_scalar("validation/commit_error", f0_commit_loss, steps)
                        if h.get('code_vq_params', None):
                            sw.add_scalar("validation/code_commit_error", code_commit_loss, steps)
                    generator.train()

            steps += 1
            if steps >= a.training_steps:
                break

        scheduler_g.step()
        scheduler_d.step()

        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))

    if rank == 0:
        print('Finished training')


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--checkpoint_path', default='cp_hifigan')
    parser.add_argument('--config', default='')
    parser.add_argument('--training_epochs', default=2000, type=int)
    parser.add_argument('--training_steps', default=90000, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=1000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--distributed-world-size', type=int)
    parser.add_argument('--distributed-port', type=int)
    parser.add_argument('--batch_size', default=-1, type=int)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    if a.batch_size > 0:
        h.batch_size = a.batch_size

    torch.manual_seed(h.seed)
    if torch.cuda.is_available() and 'WORLD_SIZE' in os.environ:
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = int(os.environ['WORLD_SIZE'])
        h.batch_size = int(h.batch_size / h.num_gpus)
        local_rank = a.local_rank
        rank = a.local_rank
        print('Batch size per GPU :', h.batch_size)
    else:
        rank = 0
        local_rank = 0

    if rank == 0:
        print("Logging:\n\ttensorboard --logdir ./checkpoints/ --samples_per_plugin=images=100000,audio=100000")

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        train(rank, local_rank, a, h)



if __name__ == '__main__':
    main()
