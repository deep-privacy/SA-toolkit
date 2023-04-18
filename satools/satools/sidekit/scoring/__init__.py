from .metric import *
from scipy.spatial.distance import cosine
import torch

from feerci import feerci

def asnorm(enroll_test_scores, enroll_xv, test_xv, cohort_xv):
    """
    Calculate adaptive s-norm

    A direct and continuous measurement of speaker confusion between all
    training samples is computationally expensive. Hence, an option is to
    compute speaker similarities from the weights of the AAM-softmax layer as
    approximations of the class-centers.

    :param enroll_test_scores: Score between enrollment and test utterances. Must be in torch.tensor format
    :param enroll_xv: Extracted xvectors from enrollment utterances. Must be in torch.tensor format
    :param test_xv: Extracted xvectors from test utterances. Must be in torch.tensor format
    :param cohort_xv: xvectors from cohort speakers. Must be in torch.tensor format
    :return: The adaptive S-Norm (ref https://www.isca-speech.org/archive/pdfs/interspeech_2017/matejka17_interspeech.pdf)
    """

    # Calculate scores between enroll and cohort
    #  print(enroll_xv.shape, cohort_xv.shape, flush=True)
    enroll_cohort_scores = torch.einsum('ij,kj', enroll_xv, cohort_xv)
    k = 200
    if cohort_xv.shape[0] < k:
        k = cohort_xv.shape[0]
    # Calculate mean and std from top 200 scores
    topk_cohort_enroll = enroll_cohort_scores.topk(k, dim=1).values
    calib_mean_enroll = topk_cohort_enroll.mean(dim=1)
    calib_std_enroll = topk_cohort_enroll.std(dim=1)

    # Calculate scores between test and cohort
    cohort_test_scores = torch.einsum('ij,kj', test_xv, cohort_xv)
    # Calculate mean and std from top 200 scores
    topk_cohort_test = cohort_test_scores.topk(k, dim=1).values
    calib_mean_test = topk_cohort_test.mean(dim=1)
    calib_std_test = topk_cohort_test.std(dim=1)

    # Calculate S-norm from z_Norm and t_norm
    z_Norm = ((enroll_test_scores - calib_mean_enroll) / calib_std_enroll)
    t_Norm = (enroll_test_scores - calib_mean_test) / calib_std_test
    return 0.5 * (z_Norm + t_Norm)


def cosine_scoring(embd1s, embd2s):
    scores = []
    for embd1, embd2 in zip(embd1s, embd2s):
        # Multiplying by -1 to ensure compatibility with affinity
        # Now lower value will indicate less affinity as compared
        # to original cosine distance
        score = 1 - cosine(embd1, embd2)
        scores.append(score)
    return scores
