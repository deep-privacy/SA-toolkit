"""
This file is part of SIDEKIT.
Copyright 2014-2023 Anthony Larcher, Pierre Champion
"""

import torch
import numpy
import math
import tqdm
import datetime
from . import scoring
from .. import utils
from .. import lr_scheduler
import logging

def train_epoch(model,
                training_opts,
                training_monitor,
                training_loader,
                optimizer,
                scheduler,
                device,
                scaler=None,
                mixed_precision=False,
                clipping=False):
    """
    Perform one epoch of training

    :param model: the Xtractor object to train
    :param training_opts: dictionary that describes the training process
    :param training_monitor: a training monitor object
    :param training_loader: the dataloader for training
    :param optimizer: an optimizer object
    :param scheduler: a scheduler object
    :param device: the device to train on
    :param scaler: boolean, if true use torch.cuda.amp.autocast
    :param clipping: boolean, if true, use gradient clipping

    :return: the Xtractor neural network avec the epoch
    """
    model.train()

    accuracy = 0.0
    running_loss = 0.0
    batch_count = 0

    if hasattr(model, 'new_epoch_hook'):
            model.new_epoch_hook(training_monitor, training_loader.dataset, scheduler)

    for batch_idx, (data, target) in enumerate(training_loader):
        data = data.squeeze().to(device)

        target = target.squeeze()
        target = target.to(device)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast('cuda', enabled=mixed_precision):
                output_tuple, _ = model(data, target=target)
                loss, cce_prediction = output_tuple
            scaler.scale(loss).backward()
            if clipping:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(optimizer)
            scaler._prev_scale = scaler.get_scale()
            scaler.update()
        else:
            output_tuple, _ = model(data, target=target)
            loss, cce_prediction = output_tuple
            loss.backward()
            if clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()

        running_loss += loss.item()
        accuracy += (torch.argmax(cce_prediction.data, 1) == target).sum().cpu()
        batch_count += 1

        if (math.fmod(batch_idx, training_opts.logging_interval) == 0) and training_opts.rank == 0:
            batch_size = target.shape[0]
            training_monitor.update(training_loss=running_loss / batch_count,
                                    training_acc=100.0 * accuracy / (batch_count*target.shape[0]),
                                    lr=scheduler._last_lr[0])

            logging.info('{} Epoch: {} [{:>6}/{} ({:>2}%)]  Loss: {:> 2.4f}  Accuracy: {:.3f}  lr={:.3e}'.format(
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                training_monitor.current_epoch+1,
                batch_idx,
                training_loader.__len__(),
                int(100. * (batch_idx+1) / training_loader.__len__()),
                running_loss / batch_count,
                100.0 * accuracy / (batch_count*target.shape[0]),
                scheduler._last_lr[0],
            ))
            running_loss = 0.0
            accuracy = 0.0
            batch_count = 0
        elif (math.fmod(batch_idx, training_opts.logging_interval//2) == 0) and training_opts.rank == 0:
            training_monitor.update(lr=scheduler._last_lr[0])
        run_scheduler(scheduler, scaler, scope="step")
    run_scheduler(scheduler, scaler, scope="epoch")
    run_scheduler(scheduler, scaler, scope="epoch", val=training_monitor.best_eer)

    return model


def run_scheduler(scheduler, scaler, scope, val=None):
    step_sch = (torch.optim.lr_scheduler.OneCycleLR, torch.optim.lr_scheduler.CyclicLR, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts, lr_scheduler.CosineAnnealingWarmRestartsWithDecayAndLinearWarmup)
    epoch_sch = (torch.optim.lr_scheduler.MultiStepLR, torch.optim.lr_scheduler.StepLR, torch.optim.lr_scheduler.ExponentialLR)
    epoch_sch_val = (torch.optim.lr_scheduler.ReduceLROnPlateau,)

    if not isinstance(scheduler, step_sch + epoch_sch + epoch_sch_val):
        raise ValueError("Scheduler '%s' not defined in  'objf.run_scheduler'" % (scheduler.__class__.__name__))

    # Call scheduler step only for batch scheduler
    if scope == "step" and isinstance(scheduler, step_sch):
        # scaler._prev_scale is to be manually added to the object!
        if scaler == None or scaler.get_scale() == scaler._prev_scale:
            scheduler.step()
    if scope == "epoch" and isinstance(scheduler, epoch_sch):
        # Call scheduler step for epoch scheduler without validation data needed
        if val==None and scaler == None or scaler.get_scale() == scaler._prev_scale:
            scheduler.step()
        # Call scheduler step when validation data is needed for epoch scheduler
        if val!=None and isinstance(scheduler, epoch_sch_val):
            if scaler == None or scaler.get_scale() == scaler._prev_scale:
                scheduler.step()



@torch.no_grad()
def validation(model, validation_loader, device, mixed_precision=False):
    """
    Perform validation during the training process

    :param model: the Xtractor object to evaluate
    :param validation_loader: dataloader object for validation
    :param device: device to compute on

    :return: a tupple with accuracy, loss and EER and EER ci_lower, ci_upper
    """

    # compute target and non-target trials from validation_set
    classes = torch.ShortTensor(validation_loader.dataset.sessions['speaker_idx'].to_numpy())
    mask = classes.unsqueeze(1) == classes.unsqueeze(1).T
    tar_indices = torch.tril(mask, -1).numpy()
    non_indices = torch.tril(~mask, -1).numpy()

    # Select a subset of non-target trials to reduce the number of tests
    tar_non_ratio = numpy.sum(tar_indices)/numpy.sum(non_indices)
    non_indices *= (numpy.random.rand(*non_indices.shape) < tar_non_ratio)


    model.eval()
    accuracy = 0.0
    loss = 0.0
    embeddings = None # torch.zeros(validation_shape)
    cursor = 0
    for batch_idx, (data, target) in enumerate(tqdm.tqdm(validation_loader, desc='validation compute')):
        if target.dim() != 1:
            target = target.squeeze()
        target = target.to(device)
        batch_size = target.shape[0]
        data = data.squeeze().to(device)
        with torch.amp.autocast('cuda', enabled=mixed_precision):
            (_loss, cce_prediction), batch_embeddings = model(data, target=target)
            accuracy += (torch.argmax(cce_prediction.data, 1) == target).sum().cpu()
            loss += _loss
        if embeddings == None:
            validation_shape = [validation_loader.dataset.__len__(), batch_embeddings.shape[-1]]
            embeddings = torch.zeros(validation_shape)
        embeddings[cursor:cursor + batch_size,:] = batch_embeddings.detach().cpu()
        cursor += batch_size

    scores = torch.einsum('ij,kj', embeddings, embeddings).cpu().numpy().astype(numpy.float64)
    negatives = scores[non_indices]
    positives = scores[tar_indices]

    feer, ci_lower, ci_upper, bootstrapped_eers = scoring.feerci(
        negatives, positives, is_sorted=False
    )

    return (100. * accuracy.numpy() / validation_shape[0],
            loss.cpu().numpy() / ((batch_idx + 1) * batch_size),
            feer*100, ci_lower*100, ci_upper*100)


@torch.no_grad()
def test(model,
         device,
         enroll_wav_scp_file,
         trials_wav_scp_file,
         enroll_utt2spk_file,
         trials_file,
         out_xvectors_and_scores,
         as_norm=True,
         mixed_precision=False,
         tqdm_file=None,
         ):
    """Compute model metrics

    :param model:
    :param device:
    :param opts: dictionary of options describing the options
    :param as_norm: boolea, if True, compute the normalized scores
    :param mixed_precision: boolea, float16 amp or not
    :param tqdm_file: str, file to log tqdm

    :return: the Equal Error Rate with bootci and min_cllr score and Normalized EER as a floats and linkability
    """
    model.eval()

    dataset_enroll = utils.WavScpDataset.from_wav_scpfile(enroll_wav_scp_file)
    dataset_trials = utils.WavScpDataset.from_wav_scpfile(trials_wav_scp_file)

    enroll_utt2spk = utils.kaldi.read_wav_scp(enroll_utt2spk_file)
    # reverse enroll_utt2spk
    enroll_spk2utt = {}
    for k, v in enroll_utt2spk.items():
        enroll_spk2utt[v] = enroll_spk2utt.get(v, []) + [k]

    with open(trials_file) as f:
        trials = [x.split() for x in f]
        utt1s = [x[0] for x in trials]
        utt2s = [x[1] for x in trials]

    ## Extract

    # !batch size = 1!
    dataloader_enroll = torch.utils.data.DataLoader(dataset_enroll, collate_fn=lambda b1:b1[0], num_workers=6)
    dataloader_trials = torch.utils.data.DataLoader(dataset_trials, collate_fn=lambda b1:b1[0], num_workers=6)

    def extract(wavinfo):
            data = wavinfo.wav.squeeze().to(device)
            with torch.amp.autocast('cuda', enabled=mixed_precision):
                _, x_vector = model(data, target=None)
            if x_vector.dim() > 1:
                x_vector = x_vector.squeeze(0)
            return x_vector

    utt2embd_enroll = {}
    utt2embd_trial = {}

    with utils.scp_io.file_writer_helper(f"scp:{out_xvectors_and_scores}/xvector.scp", overwrite=True) as f:

        for batch_idx, (wavinfo) in enumerate(tqdm.tqdm(dataloader_enroll, desc='test compute (enroll)', file=tqdm_file)):
            x_vector = extract(wavinfo).cpu()
            f[wavinfo.name] = x_vector
            utt2embd_enroll[wavinfo.name] = x_vector

        for batch_idx, (wavinfo) in enumerate(tqdm.tqdm(dataloader_trials, desc='test compute (trials)', file=tqdm_file)):
            if wavinfo.name not in utt2embd_enroll:
                x_vector = extract(wavinfo).cpu()
                f[wavinfo.name] = x_vector
                utt2embd_trial[wavinfo.name] = x_vector
            else:
                utt2embd_trial[wavinfo.name] = utt2embd_enroll[wavinfo.name]

    cohort = None
    if hasattr(model, "after_speaker_embedding") and as_norm:
        # Retrieve cohort x-vectors
        cohort = model.after_speaker_embedding.weight.data
        cohort = torch.nn.functional.normalize(cohort, dim=1).cpu()

    return compute_metrics(utt2embd_enroll, utt2embd_trial, enroll_spk2utt, trials_file, out_xvectors_and_scores, cohort=cohort)

def compute_metrics(utt2embd_enroll, utt2embd_trial, enroll_spk2utt, trials_file, out_scores, cohort=None):

    ## spk2utt xvector + Compute cosine scores

    utt2embd_enroll_mean = {}
    for spk, uttrs in enroll_spk2utt.items():
        if len(uttrs) > 1:
            mean = torch.mean(torch.stack([utt2embd_enroll[utt] for utt in uttrs]), dim=0)
            norm = torch.norm(mean, p=2)
            mean /= norm
        else:
            # don't apply l2norm twice, already done by the Xtractor
            mean = utt2embd_enroll[uttrs[0]]
        utt2embd_enroll_mean[spk] = mean


    with open(trials_file) as f:
        trials = [x.split() for x in f]
        utt1s = [x[0] for x in trials]
        utt2s = [x[1] for x in trials]

    utt2embd_enroll_mean = [utt2embd_enroll_mean[utt] for utt in utt1s]
    utt2embd_trial = [utt2embd_trial[utt] for utt in utt2s]

    scores = scoring.cosine_scoring(utt2embd_enroll_mean, utt2embd_trial)
    score_file_kaldi = []
    score_dict = {}
    for enroll, trial, score in zip(utt1s, utt2s, scores):
        score_file_kaldi.append([enroll, trial, str(score)])
        key = "_".join([enroll, trial])
        score_dict[key] = float(score)

    with open(f"{out_scores}/scores", "w") as txt_file:
        for line in score_file_kaldi:
            txt_file.write(
                " ".join(line) + "\n"
            )  # works with any number of elements in a line

    ## Compute metric

    matedScores = []
    nonMatedScores = []

    tar = []
    nontar = []
    with open(trials_file) as f:
        for i, t in enumerate(f):
            splited_t = t.split()
            key = "_".join(splited_t[:2]) # utt1s_utt2s -> score
            if splited_t[2] == "target":
                matedScores.append(score_dict[key])
                tar.append(i)
            if splited_t[2] == "nontarget":
                nonMatedScores.append(score_dict[key])
                nontar.append(i)


    matedScores = numpy.array(matedScores)
    nonMatedScores = numpy.array(nonMatedScores)

    metrics = {}

    dsys, _, _, _ = scoring.linkability(matedScores, nonMatedScores)
    cmin, matedScores, nonMatedScores = scoring.min_cllr(matedScores, nonMatedScores, return_opt=True) # return optimally calibrated scores (PAV)
    feer, ci_lower, ci_upper, bootstrapped_eers, threshold = scoring.feerci(nonMatedScores, matedScores, is_sorted=False, return_threshold=True)

    metrics["linkability"] = dsys
    metrics["eer"] = feer*100
    metrics["eer_lower"] = ci_lower*100
    metrics["eer_upper"] = ci_upper*100
    metrics["min_cllr"] = cmin
    metrics["eer_threshold"] = threshold
    metrics["asnorm"] = {}
    metrics["asnorm"]["eer"] = None
    metrics["asnorm"]["linkability"] = None
    metrics["asnorm"]["eer_lower"] = None
    metrics["asnorm"]["eer_upper"] = None
    metrics["asnorm"]["min_cllr"] = None
    metrics["asnorm"]["eer_threshold"] = None

    if cohort != None:
        enroll_test_scores = torch.FloatTensor(list(score_dict.values())).cpu()
        enroll_xv = torch.stack(utt2embd_enroll_mean)
        test_xv = torch.stack(utt2embd_trial)
        as_norm_enroll_test_scores = scoring.asnorm(enroll_test_scores, enroll_xv, test_xv, cohort)

        matedScores = as_norm_enroll_test_scores[tar].numpy().astype(numpy.float64)
        nonMatedScores = as_norm_enroll_test_scores[nontar].numpy().astype(numpy.float64)

        dsys, _, _, _ = scoring.linkability(matedScores, nonMatedScores)
        cmin, matedScores, nonMatedScores = scoring.min_cllr(matedScores, nonMatedScores, return_opt=True) # return optimally calibrated scores (PAV)
        feer, ci_lower, ci_upper, bootstrapped_eers, threshold = scoring.feerci(nonMatedScores, matedScores, is_sorted=False, return_threshold=True)

        metrics["asnorm"]["linkability"] = dsys
        metrics["asnorm"]["eer"] = feer*100
        metrics["asnorm"]["eer_lower"] = ci_lower*100
        metrics["asnorm"]["eer_upper"] = ci_upper*100
        metrics["asnorm"]["min_cllr"] = cmin
        metrics["asnorm"]["eer_threshold"] = threshold
    metrics["score"] = (matedScores, nonMatedScores)

    return metrics
