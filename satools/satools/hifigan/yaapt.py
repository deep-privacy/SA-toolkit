# -*- coding: utf-8 -*-
"""
This a ported version for PYTORRCH from the numpy/python YAAPT algorithm.
The original MATLAB program was written by Hongbing Hu and Stephen A.Zahorian.
The numpy/python was written by https://github.com/bjbschmitt.
This version is slower than the numpy/python one.
Hoover this version is version uses less cpu thread, which makes this version faster
when more than 4/5 threads are used.

The YAAPT program, designed for fundamental frequency tracking,
is extremely robust for both high quality and telephone speech.

The YAAPT program was created by the Speech Communication Laboratory of
the state university of New York at Binghamton. The original program is
available at http://www.ws.binghamton.edu/zahorian as free software. Further
information about the program could be found at Stephen A. Zahorian, and
Hongbing Hu, "A spectral/temporal method for robust fundamental frequency
tracking," J. Acoust. Soc. Am. 123(6), June 2008.

It must be noticed that, although this ported version is almost equal to the
original, some few changes were made in order to make the program more "pythonic"
and improve its performance. Nevertheless, the results obtained with both
algorithms were similar.
"""

import torch
torch.set_num_threads(1)
import torchaudio
from math import floor, ceil, isnan
from typing import Dict, List, Tuple, Any

class SignalObj(object):

    def __init__(self, data, fs:float):
        self.data = data
        self.fs = fs
        self.size = len(self.data)
        self.fs = fs
        self.new_fs = self.fs
        self.filtered = data # Nope

    def filtered_version(self, parameters:Dict[str, float]):

        # Filter the signal.

        _a = torchaudio.functional.lowpass_biquad(self.data, int(self.fs), parameters['bp_low'])
        _a = torchaudio.functional.highpass_biquad(_a, int(self.fs), parameters['bp_high'])

        # Decimate the filtered output.
        self.filtered = _a
        self.new_fs = self.fs


def medfilt(tensor, kernel_size:int):
    # Calculate padding size based on kernel size
    padding_size = kernel_size // 2
    
    # Pad the input tensor to handle edges
    padded_tensor = torch.nn.functional.pad(tensor, (padding_size,padding_size) )
    
    # Create a sliding window view of the tensor with the given kernel size
    sliding_windows = padded_tensor.unfold(0, kernel_size, 1)
    for i in range(1, tensor.dim()):
        sliding_windows = sliding_windows.unfold(i, kernel_size, 1)
    
    # Compute the median of each window along the last dimension
    medians = sliding_windows.median(dim=-1, keepdim=False)[0]
    
    return medians



"""
--------------------------------------------
                Extra functions.
--------------------------------------------
"""

def stride_matrix(vector, n_lin:int, n_col:int, hop:int) -> torch.Tensor:

    data_matrix = torch.as_strided(vector,
                               size=(n_lin, n_col),
                               stride=(vector.stride(0) * hop, vector.stride(0)))

    return data_matrix


"""
--------------------------------------------
                Classes.
--------------------------------------------
"""
"""
Auxiliary class to handle the class properties.
"""
class ClassProperty(object):

    def __init__(self, initval=None):
        self.val = initval

    def __get__(self, obj, objtype):
        return self.val

    def __set__(self, obj, val):
        self.val = val


"""
Creates a pitch object.
"""
class PitchObj(object):
    def __init__(self, frame_size:int, frame_jump:int, nfft:int=8192):
        self.nfft = nfft
        self.nframes = 0
        self.frames_pos = torch.tensor([0.0])
        self.frame_size = frame_size
        self.frame_jump = frame_jump
        self.noverlap = self.frame_size-self.frame_jump
        self.mean_energy = torch.tensor(0.0)
        self.energy = torch.tensor(0.0)
        self.vuv = torch.tensor([0.0])
        self.samp_values = torch.tensor([0.0])

    def set_energy(self, energy:torch.Tensor, threshold:float):
        self.mean_energy = torch.mean(energy)
        self.energy = energy/self.mean_energy
        self.vuv = (self.energy > threshold)

    def set_frames_pos(self, frames_pos):
        self.frames_pos = frames_pos
        self.nframes = len(self.frames_pos)

    def set_values(self, samp_values):
        self.samp_values = samp_values


"""
--------------------------------------------
                Side functions.
--------------------------------------------
"""

"""
Normalized Low Frequency Energy Ratio function. Corresponds to the nlfer.m file,
but instead of returning the results to them function, encapsulates them in the
pitch object.
"""
def nlfer(signal:SignalObj, pitch:PitchObj, parameters:Dict[str, float]):

    #---------------------------------------------------------------
    # Set parameters.
    #---------------------------------------------------------------
    N_f0_min = torch.round(torch.tensor(parameters['f0_min']*2/float(signal.new_fs))*pitch.nfft)
    N_f0_max = torch.round(torch.tensor(parameters['f0_max']/float(signal.new_fs))*pitch.nfft)

    window = torch.hann_window(pitch.frame_size+2)[1:-1]
    data = torch.zeros((signal.size))  #Needs other array, otherwise stride and
    data[:] = signal.filtered     #windowing will modify signal.filtered

    #---------------------------------------------------------------
    # Main routine.
    #---------------------------------------------------------------
    samples = torch.arange(floor(float(pitch.frame_size)/2),
                        signal.size-floor(float(pitch.frame_size)/2),
                        pitch.frame_jump)

    data_matrix = torch.empty((len(samples), pitch.frame_size))
    data_matrix[:, :] = stride_matrix(data, len(samples),
                                    pitch.frame_size, pitch.frame_jump)
    data_matrix *= window

    specData = torch.fft.rfft(data_matrix, pitch.nfft)

    frame_energy = torch.abs(specData[:, int(N_f0_min-1):int(N_f0_max)]).sum(1).to(dtype=torch.float32)
    pitch.set_energy(frame_energy, parameters['nlfer_thresh1'])
    pitch.set_frames_pos(samples)

"""
Spectral pitch tracking. Computes estimates of pitch using nonlinearly processed
speech (typically square or absolute value) and frequency domain processing.
Search for frequencies which have energy at multiplies of that frequency.
Corresponds to the spec_trk.m file.
"""
def spec_track(signal:SignalObj, pitch:PitchObj, parameters:Dict[str, float]) -> Tuple[torch.Tensor, torch.Tensor]:

    #---------------------------------------------------------------
    # Set parameters.
    #---------------------------------------------------------------
    nframe_size = pitch.frame_size*2
    maxpeaks = int(parameters['shc_maxpeaks'])
    delta = signal.new_fs/pitch.nfft

    window_length = floor(parameters['shc_window']/delta)
    half_window_length = floor(float(window_length)/2)
    if not(window_length % 2):
        window_length += 1

    max_SHC = floor((parameters['f0_max']+parameters['shc_pwidth']*2)/delta)
    min_SHC = ceil(parameters['f0_min']/delta)
    num_harmonics = int(parameters['shc_numharms'])

    #---------------------------------------------------------------
    # Main routine.
    #---------------------------------------------------------------
    cand_pitch = torch.zeros((maxpeaks, pitch.nframes))
    cand_merit = torch.ones((maxpeaks, pitch.nframes))

    data = torch.cat((signal.filtered,
                  torch.zeros(nframe_size +
                              ((pitch.nframes-1)*pitch.frame_jump-signal.size),
                              dtype=signal.filtered.dtype)),
                 dim=0)

    #Compute SHC for voiced frame
    window = torch.kaiser_window(nframe_size, periodic=True, beta=0.5)
    SHC = torch.zeros((max_SHC))
    row_mat_list = [torch.empty((max_SHC-min_SHC+1, window_length)) for _ in range(num_harmonics+1)]
    row_mat_list = torch.stack(row_mat_list)

    magnitude = torch.zeros(int((half_window_length+(pitch.nfft/2)+1)))

    _wh: List[int] = torch.where(pitch.vuv)[0].tolist()
    for frame in _wh:
        fir_step = int(frame*pitch.frame_jump)

        data_slice = data[fir_step:fir_step+nframe_size]*window
        data_slice -= torch.mean(data_slice)

        magnitude[half_window_length:] = torch.abs(torch.fft.rfft(data_slice,
                                                pitch.nfft))
        for idx,row_mat in enumerate(row_mat_list):
            row_mat[:, :] = stride_matrix(magnitude[min_SHC*(idx+1):],
                                          max_SHC-min_SHC+1,
                                          window_length, idx+1)
        SHC[min_SHC-1:max_SHC] = torch.sum(torch.prod(row_mat_list,0),1)

        frame = int(frame)
        cand_pitch[:, frame], cand_merit[:, frame] = peaks(SHC, delta, maxpeaks, parameters)


    #Extract the pitch candidates of voiced frames for the future pitch selection.
    spec_pitch = cand_pitch[0, :]
    voiced_cand_pitch = cand_pitch[:, cand_pitch[0, :] > 0.0]
    voiced_cand_merit = cand_merit[:, cand_pitch[0, :] > 0.0]
    num_voiced_cand = len(voiced_cand_pitch[0, :])
    avg_voiced = torch.mean(voiced_cand_pitch[0, :])
    std_voiced = torch.std(voiced_cand_pitch[0, :])

    #Interpolation of the weigthed candidates.
    delta1 = abs((voiced_cand_pitch - 0.8*avg_voiced))*(3-voiced_cand_merit)
    index = delta1.argmin(0)

    voiced_peak_minmrt = voiced_cand_pitch[index, list(range(num_voiced_cand))]
    voiced_merit_minmrt = voiced_cand_merit[index, list(range(num_voiced_cand))]

    voiced_peak_minmrt = medfilt(voiced_peak_minmrt,
                                 max(1, int(parameters['median_value'])-2))

    #Replace the lowest merit candidates by the median smoothed ones
    #computed from highest merit peaks above.
    voiced_cand_pitch[index, list(range(num_voiced_cand))] = voiced_peak_minmrt
    voiced_cand_merit[index, list(range(num_voiced_cand))] = voiced_merit_minmrt

    #Use dynamic programming to find best overal path among pitch candidates.
    #Dynamic weight for transition costs balance between local and
    #transition costs.
    weight_trans = parameters['dp5_k1']*std_voiced/avg_voiced

    if num_voiced_cand > 2:
        voiced_pitch = dynamic5(voiced_cand_pitch, voiced_cand_merit,
                                weight_trans, parameters['f0_min'])
        voiced_pitch = medfilt(voiced_pitch, max(1, int(parameters['median_value'])-2))

    else:
        if num_voiced_cand > 0:
            voiced_pitch = (torch.ones((num_voiced_cand)))*150.0
        else:
            voiced_pitch = torch.tensor([150.0])
            cand_pitch[0, 0] = 0

    pitch_avg = torch.mean(voiced_pitch)
    pitch_std = torch.maximum(torch.std(voiced_pitch), pitch_avg * torch.tensor(parameters['spec_pitch_min_std']))
    spec_pitch[cand_pitch[0, :] > 0] = voiced_pitch[:]

    if (spec_pitch[0] < pitch_avg/2):
        spec_pitch[0] = pitch_avg

    if (spec_pitch[-1] < pitch_avg/2):
        spec_pitch[-1] = pitch_avg


    # Find indices of non-zero values
    indices = torch.nonzero(spec_pitch).squeeze()

    # Create new spec_pitch tensor with only non-zero values
    x_new = spec_pitch[indices]

    # Create a range of indices to interpolate at
    interpolation_indices = torch.arange(spec_pitch.numel())

    # Interpolate
    spec_pitch = torch.nn.functional.interpolate(
        x_new.unsqueeze(0).unsqueeze(0), # Reshape x_new tensor
        size=(spec_pitch.numel(),),                # Interpolate to desired size
        mode='linear'
    ).squeeze()


    spec_pitch[0] = spec_pitch[2]
    spec_pitch[1] = spec_pitch[3]

    return spec_pitch, pitch_std

"""
Dynamic programming used to compute local and transition cost matrices,
enabling the lowest cost tracking of pitch candidates.
It uses NFLER from the spectrogram and the highly robust spectral F0 track,
plus the merits, for computation of the cost matrices.
Corresponds to the dynamic.m file.
"""
def dynamic(ref_pitch, ref_merit, pitch:PitchObj, parameters:Dict[str, float]):

    #---------------------------------------------------------------
    # Set parameters.
    #---------------------------------------------------------------
    num_cands = ref_pitch.shape[0]
    best_pitch = ref_pitch[num_cands-2, :]
    mean_pitch = torch.mean(best_pitch[best_pitch > 0])

    dp_w1 = parameters['dp_w1']
    dp_w2 = parameters['dp_w2']
    dp_w3 = parameters['dp_w3']
    dp_w4 = parameters['dp_w4']

    #---------------------------------------------------------------
    # Main routine.
    #---------------------------------------------------------------
    local_cost = 1 - ref_merit
    trans_cmatrix = torch.ones((num_cands, num_cands, pitch.nframes))

    ref_mat1 = torch.zeros((num_cands, num_cands, pitch.nframes))
    ref_mat2 = torch.zeros((num_cands, num_cands, pitch.nframes))
    idx_mat1 = torch.zeros((num_cands, num_cands, pitch.nframes), dtype=torch.bool)
    idx_mat2 = torch.zeros((num_cands, num_cands, pitch.nframes), dtype=torch.bool)
    idx_mat3 = torch.zeros((num_cands, num_cands, pitch.nframes), dtype=torch.bool)

    ref_mat1[:, :, 1:] = torch.tile(ref_pitch[:, 1:].reshape(1, num_cands,
                        pitch.nframes-1), (num_cands, 1, 1))
    ref_mat2[:, :, 1:] = torch.tile(ref_pitch[:, :-1].reshape(num_cands, 1,
                        pitch.nframes-1), (1, num_cands, 1))

    idx_mat1[:, :, 1:] = (ref_mat1[:, :, 1:] > 0) & (ref_mat2[:, :, 1:] > 0)
    idx_mat2[:, :, 1:] = (((ref_mat1[:, :, 1:] == 0) & (ref_mat2[:, :, 1:] > 0)) |
                       ((ref_mat1[:, :, 1:] > 0) & (ref_mat2[:, :, 1:] == 0)))
    idx_mat3[:, :, 1:] = (ref_mat1[:, :, 1:] == 0) & (ref_mat2[:, :, 1:] == 0)

    mat1_values = torch.abs(ref_mat1-ref_mat2)/mean_pitch
    benefit2 = torch.cat((torch.tensor([0]), torch.minimum(torch.tensor(1), torch.abs(pitch.energy[:-1]-pitch.energy[1:]))))
    benefit2 = benefit2.repeat(num_cands*num_cands).reshape(num_cands, num_cands, -1)


    trans_cmatrix[idx_mat1] = dp_w1*mat1_values[idx_mat1]
    trans_cmatrix[idx_mat2] = dp_w2*(1-benefit2[idx_mat2])
    trans_cmatrix[idx_mat3] = dp_w3

    trans_cmatrix = trans_cmatrix/dp_w4
    path = path1(local_cost, trans_cmatrix, num_cands, pitch.nframes)
    final_pitch = ref_pitch[path, list(range(pitch.nframes))]

    return final_pitch

"""
--------------------------------------------
                Auxiliary functions.
--------------------------------------------
"""

"""
Computes peaks in a frequency domain function associated with the peaks found
in each frame based on the correlation sequence.
Corresponds to the peaks.m file.
"""
def peaks(data, delta:float, maxpeaks:int, parameters:Dict[str, float]) -> Tuple[torch.Tensor, torch.Tensor]:

    #---------------------------------------------------------------
    # Set parameters.
    #---------------------------------------------------------------
    PEAK_THRESH1 = parameters['shc_thresh1']
    PEAK_THRESH2 = parameters['shc_thresh2']

    epsilon = .00000000000001

    width = floor(parameters['shc_pwidth']/delta)
    if not(float(width) % 2):
        width = width + 1

    center = ceil(width/2)

    min_lag = floor(parameters['f0_min']/delta - center)
    max_lag = floor(parameters['f0_max']/delta + center)

    if (min_lag < 1):
        min_lag = 1
        print('Min_lag is too low and adjusted ({}).'.format(min_lag))

    if max_lag > (len(data) - width):
        max_lag = len(data) - width
        print('Max_lag is too high and adjusted ({}).'.format(max_lag))

    pitch = torch.zeros((maxpeaks))
    merit = torch.zeros((maxpeaks))

    #---------------------------------------------------------------
    # Main routine.
    #---------------------------------------------------------------
    max_data = torch.max(data[min_lag:max_lag+1])

    if (max_data > epsilon):
        data = data/max_data

    avg_data = torch.mean(data[min_lag:max_lag+1])

    if (avg_data > 1/PEAK_THRESH1):
        pitch = torch.zeros((maxpeaks))
        merit = torch.ones((maxpeaks))
        return pitch, merit

    #---------------------------------------------------------------
    #Step1 (this step was implemented differently than in original version)
    #---------------------------------------------------------------
    numpeaks = 0
    vec_back = (data[min_lag+center+1:max_lag-center+1] >
                                            data[min_lag+center:max_lag-center])
    vec_forw = (data[min_lag+center+1:max_lag-center+1] >
                                        data[min_lag+center+2:max_lag-center+2])
    above_thresh = (data[min_lag+center+1:max_lag-center+1] >
                                        PEAK_THRESH2*avg_data)
    peaks = torch.logical_and(torch.logical_and(vec_back, vec_forw), above_thresh)

    _l: List[int] = (peaks.ravel().nonzero().flatten()+min_lag+center+1).tolist()
    for n in _l:
        n = int(n)
        if torch.argmax(data[n-center:n+center+1]) == center:
            if numpeaks >= maxpeaks:
                pitch = torch.cat((pitch, torch.zeros(1)))
                merit = torch.cat((merit, torch.zeros(1)))

            pitch[numpeaks] = float(n)*delta
            merit[numpeaks] = data[n]
            numpeaks += 1

    #---------------------------------------------------------------
    #Step2
    #---------------------------------------------------------------
    if (torch.max(merit)/avg_data < PEAK_THRESH1):
        pitch = torch.zeros((maxpeaks))
        merit = torch.ones((maxpeaks))
        return pitch, merit

    #---------------------------------------------------------------
    #Step3
    #---------------------------------------------------------------
    idx: List[int] = (-merit).ravel().argsort().tolist()
    merit = merit[idx]
    pitch = pitch[idx]

    numpeaks = min(numpeaks, maxpeaks)
    pitch = torch.cat((pitch[:numpeaks], torch.zeros(maxpeaks-numpeaks)))
    merit = torch.cat((merit[:numpeaks], torch.zeros(maxpeaks-numpeaks)))


    #---------------------------------------------------------------
    #Step4
    #---------------------------------------------------------------

    if (numpeaks > 0):
        # The first two "if pitch[0]" statements seem to had been deprecated in
        # the original YAAPT Matlab code, so they may be removed here as well.
        if (pitch[0] > parameters['f0_double']):
            numpeaks = min(numpeaks+1, maxpeaks)
            pitch[numpeaks-1] = pitch[0]/2.0
            merit[numpeaks-1] = parameters['merit_extra']

        if (pitch[0] < parameters['f0_half']):
            numpeaks = min(numpeaks+1, maxpeaks)
            pitch[numpeaks-1] = pitch[0]*2.0
            merit[numpeaks-1] = parameters['merit_extra']

        if (numpeaks < maxpeaks):
            pitch[numpeaks:maxpeaks] = pitch[0]
            merit[numpeaks:maxpeaks] = merit[0]

    else:
        pitch = torch.zeros((maxpeaks))
        merit = torch.ones((maxpeaks))

    return pitch, merit

"""
Dynamic programming used to compute local and transition cost matrices,
enabling the lowest cost tracking of pitch candidates.
It uses NFLER from the spectrogram and the highly robust spectral F0 track,
plus the merits, for computation of the cost matrices.
Corresponds to the dynamic5.m file.
"""
def dynamic5(pitch_array, merit_array, k1:float, f0_min:float):

    num_cand = pitch_array.shape[0]
    num_frames = pitch_array.shape[1]

    local = 1-merit_array
    trans = torch.zeros((num_cand, num_cand, num_frames))

    trans[:, :, 1:] = abs(pitch_array[:, 1:].reshape(1, num_cand, num_frames-1) -
                    pitch_array[:, :-1].reshape(num_cand, 1, num_frames-1))/f0_min
    trans[:, :, 1:] = 0.05*trans[:, :, 1:] + trans[:, :, 1:]**2

    trans = k1*trans
    path = path1(local, trans, num_cand, num_frames)

    final_pitch = pitch_array[path, list(range(num_frames))]

    return final_pitch

"""
Finds the optimal path with the lowest cost if two matrice(Local cost matrix
and Transition cost) are given.
Corresponds to the path1.m file.
"""
def path1(local, trans, n_lin:int, n_col:int):

# Apparently the following lines are somehow kind of useless.
# Therefore, I removed them in the version 1.0.3.

#    if n_lin >= 100:
#        print 'Stop in Dynamic due to M>100'
#        raise KeyboardInterrupt
#
#    if n_col >= 1000:
#        print 'Stop in Dynamic due to N>1000'
#        raise KeyboardInterrupt

    PRED = torch.zeros((n_lin, n_col), dtype=torch.long)
    P = torch.ones((n_col), dtype=torch.long)
    p_small = torch.zeros((n_col), dtype=torch.long)

    PCOST = torch.zeros((n_lin))
    CCOST = torch.zeros((n_lin))
    PCOST = local[:, 0]

    for I in range(1, n_col):

        aux_matrix = PCOST+trans[:, :, I]
        K = n_lin-torch.argmin(torch.flip(aux_matrix, [1]), 1)-1
        PRED[:, I] = K
        CCOST = PCOST[K]+trans[K, list(range(n_lin)), I]

        assert CCOST.any() < 1.0E+30, 'CCOST>1.0E+30, Stop in Dynamic'
        CCOST = CCOST+local[:, I]

        PCOST[:] = CCOST
        J = n_lin - torch.argmin(torch.flip(CCOST, dims=[0]), dim=0) - 1
        p_small[I] = J

    P[-1] = p_small[-1]

    for I in range(n_col-2, -1, -1):
        P[I] = PRED[P[I+1], I+1]

    return P

"""
Computes the NCCF (Normalized cross correlation Function) sequence based on
the RAPT algorithm discussed by DAVID TALKIN.
Corresponds to the crs_corr.m file.
"""
def crs_corr(data, lag_min:int, lag_max:int):

    eps1 = 0.0
    data_len = len(data)
    N = data_len-lag_max

    error_str = 'ERROR: Negative index in the cross correlation calculation of '
    error_str += 'the pYAAPT time domain analysis. Please try to increase the '
    error_str += 'value of the "tda_frame_length" parameter.'
    assert N>0, error_str

    phi = torch.zeros((data_len))
    data -= torch.mean(data)
    x_j = data[0:N]
    x_jr = data[lag_min:lag_max+N]
    p = torch.dot(x_j, x_j)

    x_jr_matrix = stride_matrix(x_jr, lag_max-lag_min, N, 1)

    x_j = x_j.unsqueeze(0).T
    formula_nume = torch.matmul(x_jr_matrix, x_j).squeeze()
    formula_denom = torch.sum(x_jr_matrix*x_jr_matrix, 1)*p + eps1

    phi[lag_min:lag_max] = formula_nume/torch.sqrt(formula_denom)

    return phi

"""
Computes pitch estimates and the corresponding merit values associated with the
peaks found in each frame based on the correlation sequence.
Corresponds to the cmp_rate.m file.
"""
def cmp_rate(phi, fs:float, maxcands:int, lag_min:int, lag_max:int, parameters:Dict[str, float]) -> Tuple[torch.Tensor, torch.Tensor]:

    #---------------------------------------------------------------
    # Set parameters.
    #---------------------------------------------------------------
    width = parameters['nccf_pwidth']
    center = floor(width/2.0)
    merit_thresh1 = parameters['nccf_thresh1']
    merit_thresh2 = parameters['nccf_thresh2']

    numpeaks = 0
    pitch = torch.zeros((maxcands))
    merit = torch.zeros((maxcands))

    #---------------------------------------------------------------
    # Main routine.
    #(this step was implemented differently than in original version)
    #---------------------------------------------------------------
    vec_back = (phi[lag_min+center:lag_max-center+1] >
                                            phi[lag_min+center-1:lag_max-center])
    vec_forw = (phi[lag_min+center:lag_max-center+1] >
                                        phi[lag_min+center+1:lag_max-center+2])
    above_thresh = phi[lag_min+center:lag_max-center+1] > merit_thresh1
    peaks = torch.logical_and(torch.logical_and(vec_back, vec_forw), above_thresh)

    if peaks.ravel().nonzero().shape[0] != 0:
        _peaks = (peaks.ravel().nonzero()[0]+lag_min+center)
        
        if torch.amax(phi) > merit_thresh2 and len(_peaks) > 0:
            mask = phi[_peaks]
            max_point = _peaks[torch.argmax(mask)]
            pitch[numpeaks] = fs/float(max_point+1)
            merit[numpeaks] = torch.amax(phi[_peaks])
            numpeaks += 1
        else:
            for n in _peaks:
                if torch.argmax(phi[n-center:n+center+1]) == center:
                    pitch[numpeaks] = fs/float(n+1)
                    merit[numpeaks] = phi[n]
    else:
        for n in peaks:
            if n == torch.tensor(False):
                continue
            if torch.argmax(phi[n-center:n+center+1]) == center:
                pitch[numpeaks] = fs/float(n+1)
                merit[numpeaks] = phi[n]
                #  try:
                    #  pitch[numpeaks] = fs/float(n+1)
                    #  merit[numpeaks] = phi[n]
                #  except:
                    #  pitch = torch.hstack((pitch, fs/float(n+1)))
                    #  merit = torch.hstack((merit, phi[n]))
                numpeaks += 1

    #---------------------------------------------------------------
    # Sort the results.
    #---------------------------------------------------------------
    idx: List[int] = (-merit).ravel().argsort().tolist()
    merit = merit[idx[:maxcands]]
    pitch = pitch[idx[:maxcands]]

    if (torch.amax(merit) > 1.0):
        merit = merit/torch.amax(merit)

    return pitch, merit


"""
Temporal pitch tracking.
Corresponds to the tm_trk.m file.
"""
@torch.jit.script
def time_track(signal:SignalObj, spec_pitch, pitch_std, pitch:PitchObj, parameters:Dict[str, float]) -> Tuple[torch.Tensor, torch.Tensor]:

    #---------------------------------------------------------------
    # Set parameters.
    #---------------------------------------------------------------
    tda_frame_length = int(parameters['tda_frame_length']*signal.fs/1000)
    tda_noverlap = tda_frame_length-pitch.frame_jump
    tda_nframes = int((len(signal.data)-tda_noverlap)/pitch.frame_jump)

    len_spectral = len(spec_pitch)
    if tda_nframes < len_spectral:
        spec_pitch = spec_pitch[:tda_nframes]
    elif tda_nframes > len_spectral:
        tda_nframes = len_spectral

    merit_boost = parameters['merit_boost']
    maxcands = int(parameters['nccf_maxcands'])
    freq_thresh = 5.0*pitch_std

    spec_range = torch.max(spec_pitch - 2.0 * pitch_std, torch.tensor(parameters['f0_min']))
    spec_range = torch.vstack((spec_range, torch.min(spec_pitch + 2.0 * pitch_std, torch.tensor(parameters['f0_max']))))

    time_pitch = torch.zeros((maxcands, tda_nframes))
    time_merit = torch.zeros((maxcands, tda_nframes))

    #---------------------------------------------------------------
    # Main routine.
    #---------------------------------------------------------------
    data = torch.zeros((signal.size))  #Needs other array, otherwise stride and
    data[:] = signal.filtered       #windowing will modify signal.filtered
    signal_frames = stride_matrix(data, tda_nframes,tda_frame_length,
                                  pitch.frame_jump)
    for frame in range(tda_nframes):
        a = floor(signal.new_fs/spec_range[1, frame])
        b = floor(signal.new_fs/spec_range[0, frame])
        if not isnan(a) and not isnan(b):
            lag_min0 = int(a - floor(parameters['nccf_pwidth']/2.0))
            lag_max0 = int(b + floor(parameters['nccf_pwidth']/2.0))

            phi = crs_corr(signal_frames[frame, :], lag_min0, lag_max0)
            time_pitch[:, frame], time_merit[:, frame] = \
                cmp_rate(phi, signal.new_fs, maxcands, lag_min0, lag_max0, parameters)

    diff = torch.abs(time_pitch - spec_pitch)
    match1 = (diff < freq_thresh)
    match = ((1 - diff/freq_thresh) * match1)
    time_merit = (((1+merit_boost)*time_merit) * match)

    return time_pitch, time_merit


def refine(time_pitch1, time_merit1, time_pitch2, time_merit2, spec_pitch,
           pitch:PitchObj, parameters:Dict[str, float]) -> Tuple[torch.Tensor, torch.Tensor]:

    #---------------------------------------------------------------
    # Set parameters.
    #---------------------------------------------------------------
    nlfer_thresh2 = parameters['nlfer_thresh2']
    merit_pivot = parameters['merit_pivot']

    #---------------------------------------------------------------
    # Main routine.
    #---------------------------------------------------------------
    time_pitch = torch.cat((time_pitch1, time_pitch2), 0)
    time_merit = torch.cat((time_merit1, time_merit2), 0)
    maxcands = time_pitch.shape[0]

    idx = torch.argsort(-time_merit, dim=0)
    time_merit, _ = torch.sort(time_merit, dim=0)
    time_merit[:, :] = torch.flip(time_merit, dims=[0])

    time_pitch = time_pitch[idx, torch.arange(pitch.nframes)]

    best_pitch = medfilt(time_pitch[0, :], int(parameters['median_value']))*pitch.vuv

    idx1 = pitch.energy <= nlfer_thresh2
    idx2 = (pitch.energy > nlfer_thresh2) & (time_pitch[0, :] > 0)
    idx3 = (pitch.energy > nlfer_thresh2) & (time_pitch[0, :] <= 0)
    merit_mat = (time_pitch[1:maxcands-1, :] == 0) & idx2
    merit_mat = torch.cat((torch.zeros((1, pitch.nframes), dtype=torch.bool),
                          merit_mat, torch.zeros((1, pitch.nframes), dtype=torch.bool)), 0)

    time_pitch[:, idx1] = 0
    time_merit[:, idx1] = merit_pivot

    time_pitch[maxcands-1, idx2] = 0.0
    time_merit[maxcands-1, idx2] = 1.0-time_merit[0, idx2]
    time_merit[merit_mat] = 0.0

    time_pitch[0, idx3] = spec_pitch[idx3]
    time_merit[0, idx3] = torch.minimum(torch.tensor(1), pitch.energy[idx3]/2.0)
    time_pitch[1:maxcands, idx3] = 0.0
    time_merit[1:maxcands, idx3] = 1.0-time_merit[0, idx3]

    time_pitch[maxcands-2, :] = best_pitch
    non_zero_frames = best_pitch > 0.0
    time_merit[maxcands-2, non_zero_frames] = time_merit[0, non_zero_frames]
    time_merit[maxcands-2, ~(non_zero_frames)] = 1.0-torch.minimum(torch.tensor(1),
                                       pitch.energy[~(non_zero_frames)]/2.0)

    time_pitch[maxcands-3, :] = spec_pitch
    time_merit[maxcands-3, :] = pitch.energy/5.0

    return time_pitch, time_merit




"""
--------------------------------------------
                Main function.
--------------------------------------------
"""
@torch.no_grad()
@torch.jit.script
def yaapt(_in:torch.Tensor, kwargs:Dict[str, float]):

    # Rename the YAAPT v4.0 parameter "frame_lengtht" to "tda_frame_length"
    # (if provided).
    if 'frame_lengtht' in kwargs:
        if 'tda_frame_length' in kwargs:
            warning_str = 'WARNING: Both "tda_frame_length" and "frame_lengtht" '
            warning_str += 'refer to the same parameter. Therefore, the value '
            warning_str += 'of "frame_lengtht" is going to be discarded.'
            print(warning_str)
        else:
            kwargs['tda_frame_length'] = kwargs.pop('frame_lengtht')

    #---------------------------------------------------------------
    # Set the default values for the parameters.
    #---------------------------------------------------------------
    parameters:Dict[str, float] = {}
    parameters['sr'] = kwargs.get('sr', 16000.0)   #sampling rate
    parameters['frame_length'] = kwargs.get('frame_length', 35.0)   #Length of each analysis frame (ms)
    # WARNING: In the original MATLAB YAAPT 4.0 code the next parameter is called
    # "frame_lengtht" which is quite similar to the previous one "frame_length".
    # Therefore, I've decided to rename it to "tda_frame_length" in order to
    # avoid confusion between them. Nevertheless, both inputs ("frame_lengtht"
    # and "tda_frame_length") are accepted when the function is called.
    parameters['tda_frame_length'] = \
                              kwargs.get('tda_frame_length', 35.0)  #Frame length employed in the time domain analysis (ms)
    parameters['frame_space'] = kwargs.get('frame_space', 10.0)     #Spacing between analysis frames (ms)
    parameters['f0_min'] = kwargs.get('f0_min', 60.0)               #Minimum F0 searched (Hz)
    parameters['f0_max'] = kwargs.get('f0_max', 400.0)              #Maximum F0 searched (Hz)
    parameters['fft_length'] = kwargs.get('fft_length', 8192.0)       #FFT length
    parameters['bp_low'] = kwargs.get('bp_low', 50.0)               #Low frequency of filter passband (Hz)
    parameters['bp_high'] = kwargs.get('bp_high', 1500.0)           #High frequency of filter passband (Hz)
    parameters['nlfer_thresh1'] = kwargs.get('nlfer_thresh1', 0.75) #NLFER boundary for voiced/unvoiced decisions
    parameters['nlfer_thresh2'] = kwargs.get('nlfer_thresh2', 0.1)  #Threshold for NLFER definitely unvoiced
    parameters['shc_numharms'] = kwargs.get('shc_numharms', 3.0)      #Number of harmonics in SHC calculation
    parameters['shc_window'] = kwargs.get('shc_window', 40.0)       #SHC window length (Hz)
    parameters['shc_maxpeaks'] = kwargs.get('shc_maxpeaks', 4.0)      #Maximum number of SHC peaks to be found
    parameters['shc_pwidth'] = kwargs.get('shc_pwidth', 50.0)       #Window width in SHC peak picking (Hz)
    parameters['shc_thresh1'] = kwargs.get('shc_thresh1', 5.0)      #Threshold 1 for SHC peak picking
    parameters['shc_thresh2'] = kwargs.get('shc_thresh2', 1.25)     #Threshold 2 for SHC peak picking
    parameters['f0_double'] = kwargs.get('f0_double', 150.0)        #F0 doubling decision threshold (Hz)
    parameters['f0_half'] = kwargs.get('f0_half', 150.0)            #F0 halving decision threshold (Hz)
    parameters['dp5_k1'] = kwargs.get('dp5_k1', 11.0)               #Weight used in dynamic program
    parameters['nccf_thresh1'] = kwargs.get('nccf_thresh1', 0.3)    #Threshold for considering a peak in NCCF
    parameters['nccf_thresh2'] = kwargs.get('nccf_thresh2', 0.9)    #Threshold for terminating serach in NCCF
    parameters['nccf_maxcands'] = kwargs.get('nccf_maxcands', 3.0)    #Maximum number of candidates found
    parameters['nccf_pwidth'] = kwargs.get('nccf_pwidth', 5.0)        #Window width in NCCF peak picking
    parameters['merit_boost'] = kwargs.get('merit_boost', 0.20)     #Boost merit
    parameters['merit_pivot'] = kwargs.get('merit_pivot', 0.99)     #Merit assigned to unvoiced candidates in
                                                                    #defintely unvoiced frames
    parameters['merit_extra'] = kwargs.get('merit_extra', 0.4)      #Merit assigned to extra candidates
                                                                    #in reducing F0 doubling/halving errors
    parameters['median_value'] = kwargs.get('median_value', 7.0)      #Order of medial filter
    parameters['dp_w1'] = kwargs.get('dp_w1', 0.15)                 #DP weight factor for V-V transitions
    parameters['dp_w2'] = kwargs.get('dp_w2', 0.5)                  #DP weight factor for V-UV or UV-V transitions
    parameters['dp_w3'] = kwargs.get('dp_w3', 0.1)                  #DP weight factor of UV-UV transitions
    parameters['dp_w4'] = kwargs.get('dp_w4', 0.9)                  #Weight factor for local costs

    # Exclusive from pYAAPT.

    parameters['spec_pitch_min_std'] = kwargs.get('spec_pitch_min_std', 0.05) 
                                                                    #Weight factor that sets a minimum
                                                                    #spectral pitch standard deviation,
                                                                    #which is calculated as 
                                                                    #min_std = pitch_avg*spec_pitch_min_std

    #---------------------------------------------------------------
    # Create the signal objects and filter them.
    #---------------------------------------------------------------
    to_pad = int(parameters["frame_length"] / 1000 * int(parameters["sr"])) // 2
    _in = torch.nn.functional.pad(_in.squeeze(), (to_pad, to_pad))
    signal = SignalObj(_in, parameters["sr"])

    nonlinear_sign = SignalObj(signal.data**2, parameters["sr"])

    signal.filtered_version(parameters)
    nonlinear_sign.filtered_version(parameters)

    #---------------------------------------------------------------
    # Create the pitch object.
    #---------------------------------------------------------------
    nfft = int(parameters['fft_length'])
    frame_size = floor(torch.tensor(parameters['frame_length']*signal.fs/1000))
    frame_jump = floor(torch.tensor(parameters['frame_space']*signal.fs/1000))
    pitch = PitchObj(int(frame_size), int(frame_jump), int(nfft))

    assert pitch.frame_size > 15, 'Frame length value {} is too short.'.format(pitch.frame_size)
    assert pitch.frame_size < 2048, 'Frame length value {} exceeds the limit.'.format(pitch.frame_size)


    #---------------------------------------------------------------
    # Calculate NLFER and determine voiced/unvoiced frames.
    #---------------------------------------------------------------
    nlfer(signal, pitch, parameters)

    #---------------------------------------------------------------
    # Calculate an approximate pitch track from the spectrum.
    #---------------------------------------------------------------
    spec_pitch, pitch_std = spec_track(nonlinear_sign, pitch, parameters)

    #---------------------------------------------------------------
    # Temporal pitch tracking based on NCCF.
    #---------------------------------------------------------------
    fut = torch.jit.fork(time_track, signal, spec_pitch, pitch_std, pitch,
                                          parameters)

    fut1 = torch.jit.fork(time_track, nonlinear_sign, spec_pitch, pitch_std,
                                          pitch, parameters)
    time_pitch1, time_merit1 = torch.jit.wait(fut)
    time_pitch2, time_merit2 = torch.jit.wait(fut1)

    #  time_pitch1, time_merit1 = time_track(signal, spec_pitch, pitch_std, pitch,
                                          #  parameters)

    #  time_pitch2, time_merit2 = time_track(nonlinear_sign, spec_pitch, pitch_std,
                                          #  pitch, parameters)


    # Added in YAAPT 4.0
    if time_pitch1.shape[1] < len(spec_pitch):
        len_time = time_pitch1.shape[1]
        len_spec = len(spec_pitch)
        time_pitch1 = torch.concatenate((time_pitch1, torch.zeros((3,len_spec-len_time),
                                      dtype=time_pitch1.dtype)),1)
        time_pitch2 = torch.concatenate((time_pitch2, torch.zeros((3,len_spec-len_time),
                                      dtype=time_pitch2.dtype)),1)
        time_merit1 = torch.concatenate((time_merit1, torch.zeros((3,len_spec-len_time),
                                      dtype=time_merit1.dtype)),1)
        time_merit2 = torch.concatenate((time_merit2, torch.zeros((3,len_spec-len_time),
                                      dtype=time_merit2.dtype)),1)

    #---------------------------------------------------------------
    # Refine pitch candidates.
    #---------------------------------------------------------------
    ref_pitch, ref_merit = refine(time_pitch1, time_merit1, time_pitch2,
                                  time_merit2, spec_pitch, pitch, parameters)

    #---------------------------------------------------------------
    # Use dyanamic programming to determine the final pitch.
    #---------------------------------------------------------------
    final_pitch = dynamic(ref_pitch, ref_merit, pitch, parameters)

    pitch.set_values(final_pitch)

    return pitch

if __name__ == "__main__":
    import torchaudio
    import librosa
    wav, _ = torchaudio.load("https://datasets-server.huggingface.co/assets/librispeech_asr/--/all/train.other.500/1/audio/audio.mp3")
    audio = wav.squeeze()

    _yaapt_opts = {
        "frame_length": 35.0,
        "frame_space": 20.0,
        "nccf_thresh1": 0.25,
        "tda_frame_length": 25.0,
    }

    pitch = yaapt(
        audio,
        _yaapt_opts,
    )
    print(pitch.samp_values)

    def eval():
        pitch = yaapt(
            audio,
            _yaapt_opts,
        )

    def timed(fn):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        torch.cuda.synchronize()
        return result, start.elapsed_time(end) / 1000


    N_ITERS = 100
    eager_times = []
    compile_times = []
    for i in range(N_ITERS):
        _, eager_time = timed(lambda: eval())
        eager_times.append(eager_time)
        print(f"eval time {i}: {eager_time}")

    import numpy as np
    eager_med = np.median(eager_times)
    print("~" * 10)
    print("Time per opts", eager_med)
