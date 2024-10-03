# FROM https://gitlab.inria.fr/magnet/anonymization_metrics.git, see LICENSE.txt
import numpy as np
from numpy import isinf, zeros, exp, log, hstack, vstack, finfo, array, inf, argwhere, isscalar
from numpy import linspace, minimum, maximum, repeat, flipud, abs
import pandas as pd
import copy



def linkability(matedScores, nonMatedScores, omega=1,nBins=-1):
    """Compute Linkability measure between mated
    and non-mated scores.

    Parameters
    ----------
    matedScores : Array_like
        List of scores associated to mated pairs
    nonMatedScores : Array_like
        List of scores associated to non-mated pairs
    omega : float
        Prior ration P[mated]/P[non-mated]

    Returns
    -------
    Dsys : float
        Global linkability measure.
    D : ndarray
        Local linkability measure for each bin.
    bin_centers : ndarray
        Center of the bins (from historgrams).
    bin_edges : ndarray
        Edges of the bis (from histograms)

    Notes
    -----
    Adaptation of the linkability measure of Gomez-Barrero et al. [1]

    References
    ----------

    .. [1] Gomez-Barrero, M., Galbally, J., Rathgeb, C. and Busch,
    C., 2017. General framework to evaluate unlinkability in biometric
    template protection systems. IEEE Transactions on Information
    Forensics and Security, 13(6), pp.1406-1420.
    """
    # Limiting the number of bins (100 maximum or lower if few scores available)
    if nBins < 0:
        nBins = min(int(len(matedScores) / 10), 100)

    # define range of scores to compute D
    bin_edges=np.linspace(min([min(matedScores), min(nonMatedScores)]),  max([max(matedScores), max(nonMatedScores)]), num=nBins + 1, endpoint=True)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    # compute score distributions using normalized histograms
    y1 = np.histogram(matedScores, bins = bin_edges, density = True)[0]
    y2 = np.histogram(nonMatedScores, bins = bin_edges, density = True)[0]
    # LR = P[s|mated ]/P[s|non-mated]
    LR = np.divide(y1, y2, out=np.ones_like(y1), where=y2!=0)
    # compute D
    D = 2*(omega*LR/(1 + omega*LR)) - 1
    # Def of D
    D[omega*LR <= 1] = 0
    # Taking care of inf/NaN
    mask =  [True if y2[i]==0 and y1[i]!=0 else False for i in range(len(y1))]
    D[mask] = 1
    # Global measure using trapz numerical integration
    Dsys = np.trapz(x = bin_centers, y = D* y1)
    return Dsys, D, bin_centers, bin_edges



def llr_from_bins(matedScores, nonMatedScores, nBins=0):
    """Estimation of Log-likelihood ratios (LRR) using
    descretized bins.

    It computes  log(P[s|mated]/P[s|non-mated])
    by estimating P[s|mated] using histograms


    Parameters
    ----------
    matedScores : Array_like
        List of scores associated to mated pairs
    nonMatedScores : Array_like
        List of scores associated to non-mated pairs
    nBins : Int, Optional
        Number of Bins, default automated

    Returns
    -------
    mated_llrs : ndarray
        LRRs associated to the input mated scores.
    nonmated_llrs : ndarray
        LRRs associated to the input non-mated scores.
    """
    if nBins == 0 :
      # Limiting the number of bins
      #(100 maximum or lower if few scores available)
    	nBins = min(int(len(matedScores) / 2), 100)
    # Generating the bins
    maxS=max([max(matedScores), max(nonMatedScores)])
    minS=min([min(matedScores), min(nonMatedScores)])
    bin_edges=np.linspace(minS,maxS, num=nBins + 1, endpoint=True)
    # Estimating P[s|mated] and P[s|non-mated]
    y1 = np.histogram(matedScores, bins = bin_edges, density = True)[0]
    y2 = np.histogram(nonMatedScores, bins = bin_edges, density = True)[0]
    # LR = P[s|mated ]/P[s|non-mated]
    LR = np.divide(y1, y2, out=np.ones_like(y1), where=y2!=0)
    LLR = np.log(LR)
    # Function to extract the index of the correct bin for a score
    def firstGreaterIndex(tab,value):
    	return next(x[0] for x in enumerate(tab) if x[1] > value)
    # Associated each score to the LLR of its bin
    mated_llrs= [ LLR[firstGreaterIndex(bin_edges,s)-1] if s!=maxS else len(bin_edges)-1 for s in matedScores ]
    nonmated_llrs= [ LLR[firstGreaterIndex(bin_edges,s)-1] if s!=maxS else len(bin_edges)-1 for s in nonMatedScores ]
    return np.array(mated_llrs), np.array(nonmated_llrs)


def draw_scores(matedScores, nonMatedScores, Dsys, D, bin_centers, bin_edges, output_file):
    """Draw both mated and non-mated score distributions
    and associated their associated local linkability

    Parameters
    ----------
    matedScores : Array_like
        list of scores associated to mated pairs
    nonMatedScores : Array_like
        list of scores associated to non-mated pairs
    Dsys : float
        Global linkability measure.
    D : ndarray
        Local linkability measure for each bin.
    bin_centers : ndarray
        Center of the bins (from historgrams).
    bin_edges : ndarray
        Edges of the bis (from histograms)
    output_file : String
        Path to png and pdf output file.

    References
    ----------

    .. [1] Gomez-Barrero, M., Galbally, J., Rathgeb, C., & Busch,
     C. (2017). General framework to evaluate unlinkability in
     biometric template protection systems. IEEE Transactions
     on Information Forensics and Security, 13(6), 1406-1420.
    """
    import seaborn as sns
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.legend import Legend

    # Colorblind and photocopy friendly colors
    colors=['#e66101','#fdb863','#b2abd2','#5e3c99']
    legendLocation='upper left'
    plt.clf()
    # Kernel density estimate of the score
    ax = sns.kdeplot(matedScores, shade=False, label='Same Speaker', color=colors[2],linewidth=2, linestyle='--')
    x1,y1 = ax.get_lines()[0].get_data()
    ax = sns.kdeplot(nonMatedScores, shade=False, label='Not Same Speaker', color=colors[0],linewidth=2, linestyle=':')
    x2,y2 = ax.get_lines()[1].get_data()
    # Associated local linkability
    ax2 = ax.twinx()
    lns3, = ax2.plot(bin_centers, D, label='$\mathrm{D}_{\leftrightarrow}(s)$', color=colors[3],linewidth=2)

    # #print omega * LR = 1 lines
    index = np.where(D <= 0)
    ax.axvline(bin_centers[index[0][0]], color='k', linestyle='--')

    # Figure formatting
    ax.spines['top'].set_visible(False)
    ax.set_ylabel("Probability Density")
    ax.set_xlabel("Score")
    # Global Linkability
    ax.set_title("$\mathrm{D}_{\leftrightarrow}^{\mathit{sys}}$ = %.2f" % (Dsys),  y = 1.02)
    # Legends
    labs = [ax.get_lines()[0].get_label(), ax.get_lines()[1].get_label(), ax2.get_lines()[0].get_label()]
    lns = [ax.get_lines()[0], ax.get_lines()[1], lns3]
    ax.legend(lns, labs, loc = legendLocation)
    # Frame of Values
    ax.set_ylim([0, max(max(y1), max(y2)) * 1.05])
    ax.set_xlim([bin_edges[0]*0.98, bin_edges[-1]*1.02])
    ax2.set_ylim([0, 1.1])
    ax2.set_ylabel("$\mathrm{D}_{\leftrightarrow}(s)$")

    # Optional: getting rid of possible extensions
    outname= output_file.replace('.pdf', '').replace('.png', '').replace('.csv', '').replace('.txt', '')
    plt.savefig(outname + ".pdf", format="pdf")
    plt.savefig(outname + ".png", format="png")



def logit(p):
    """logit function of each element of an array.
    It computes the log-odds associated to a probability
    Mapping [0,1] --> real line
    logit(p) = log(p/(1-p))
    Its inverse is the sigmoid function
    Parameters
    ----------
    p : Array_like
        A list of probabilities in [0,1]

    Returns
    -------
    lp : ndarray
        list of associated log-odds.
     """
    p = np.array(p)
    lp = np.zeros(p.shape)
    f0 = p == 0
    f1 = p == 1
    f = (p > 0) & (p < 1)

    if lp.shape == ():
        if f:
            lp = np.log(p / (1 - p))
        elif f0:
            lp = -np.inf
        elif f1:
            lp = np.inf
    else:
        lp[f] = np.log(p[f] / (1 - p[f]))
        lp[f0] = -np.inf
        lp[f1] = np.inf
    return lp


def sigmoid(log_odds):
    """sigmoid function of each element of an array.
    It computes the probability associated to a log-odd
    Mapping real line --> [0,1]
    sigmoid(x) = 1/(1+exp(-x))
    Its inverse is the logit function
    Parameters
    ----------
    p : Array_like
        A list of log-odds

    Returns
    -------
    lp : ndarray
        list of associated probabilities.
     """
    p = 1 / (1 + np.exp(-log_odds))
    return p


def cllr(tar_llrs, nontar_llrs):
    """Computes the application-independent cost function
    It is an expected error of a binary decision
    based on the target and non-target scores (mated/non-mated)
    The higher wrost is the average error
    Parameters
    ----------
    tar_llrs : ndarray
        list of scores associated to target pairs
        (uncalibrated LLRs)
    nontar_llrs : ndarray
        list of scores associated to non-target pairs
        (uncalibrated LLRs)

    Returns
    -------
    c : float
        Cllr of the scores.

    Notes
    -----
    Adaptation of the cllr measure of Brümmer et al. [2]
    Credits to Andreas Nautsch (EURECOM)
    https://gitlab.eurecom.fr/nautsch/cllr


    References
    ----------

    .. [2] Brümmer, N., & Du Preez, J. (2006).
    Application-independent evaluation of speaker detection.
    Computer Speech & Language, 20(2-3), 230-275.
    """
    log2 = np.log(2)
    tar_posterior = sigmoid(tar_llrs)
    non_posterior = sigmoid(-nontar_llrs)
    if any(tar_posterior == 0) or any(non_posterior == 0):
        return np.inf

    c1 = (-np.log(tar_posterior)).mean() / log2
    c2 = (-np.log(non_posterior)).mean() / log2
    c = (c1 + c2) / 2
    return c


def min_cllr(tar_llrs, nontar_llrs, monotonicity_epsilon=1e-6, compute_eer=False, return_opt=False):
    """Computes the minimum application-independent cost function
    under calibrated scores (LLR)
    It is an expected error of a binary decision
    based on the target and non-target scores (mated/non-mated)
    The higher wrost is the average error
    Parameters
    ----------
    tar_llrs : ndarray
        list of scores associated to target pairs
        (uncalibrated LLRs)
    nontar_llrs : ndarray
        list of scores associated to non-target pairs
        (uncalibrated LLRs)
    monotonicity_epsilon : float
        Unsures monoticity of the optimal LLRs
    compute_eer : bool
        Returns ROCCH-EER
    return_opt : bool
        Returns optimal scores

    Returns
    -------
    cmin : float
        minCllr of the scores.
    eer : float
        ROCCH-EER
    tar : ndarray
        Target optimally calibrated scores (PAV)
    non : ndarray
        Non-target optimally calibrated scores (PAV)

    Notes
    -----
    Adaptation of the cllr measure of Brümmer et al. [2]
    Credits to Andreas Nautsch (EURECOM)
    https://gitlab.eurecom.fr/nautsch/cllr


    References
    ----------

    .. [2] Brümmer, N., & Du Preez, J. (2006).
    Application-independent evaluation of speaker detection.
    Computer Speech & Language, 20(2-3), 230-275.
    """
    if compute_eer:
        [tar, non, eer] = optimal_llr(tar_llrs, nontar_llrs, laplace=False, monotonicity_epsilon=monotonicity_epsilon,
                                     compute_eer=compute_eer)
        cmin = cllr(tar, non)
        if not return_opt:
            return cmin, eer
        else:
            return cmin, eer, tar, non
    else:
        [tar,non] = optimal_llr(tar_llrs, nontar_llrs, laplace=False, monotonicity_epsilon=monotonicity_epsilon)
        cmin = cllr(tar, non)
        if not return_opt:
            return cmin
        else:
            return cmin, tar, non



def pavx(y):
    """PAV: Pool Adjacent Violators algorithm
    With respect to an input vector v, it computes ghat
    a nondecreasing vector such as sum((y - ghat).^2) is minimal

    Parameters
    ----------
    y : ndarray
        Input vector

    Returns
    -------
    ghat : ndarray
        output array.
    width : ndarray
        Width of bins generated by the PAV
    height : ndarray
        Height of bins generated by the PAV


    Notes
    -----
    Credits to Andreas Nautsch (EURECOM)
    https://gitlab.eurecom.fr/nautsch/cllr

    References
    ----------

    .. [2] Brümmer, N., & Du Preez, J. (2006).
    Application-independent evaluation of speaker detection.
    Computer Speech & Language, 20(2-3), 230-275.
    """
    assert y.ndim == 1, 'Argument should be a 1-D array'
    assert y.shape[0] > 0, 'Input array is empty'
    n = y.shape[0]

    index = np.zeros(n, dtype=int)
    length = np.zeros(n, dtype=int)

    ghat = np.zeros(n)

    ci = 0
    index[ci] = 1
    length[ci] = 1
    ghat[ci] = y[0]

    for j in range(1, n):
        ci += 1
        index[ci] = j + 1
        length[ci] = 1
        ghat[ci] = y[j]
        while (ci >= 1) & (ghat[np.max(ci - 1, 0)] >= ghat[ci]):
            nw = length[ci - 1] + length[ci]
            ghat[ci - 1] = ghat[ci - 1] + (length[ci] / nw) * (ghat[ci] - ghat[ci - 1])
            length[ci - 1] = nw
            ci -= 1

    height = copy.deepcopy(ghat[:ci + 1])
    width = copy.deepcopy(length[:ci + 1])

    while n >= 0:
        for j in range(index[ci], n + 1):
            ghat[j - 1] = ghat[ci]
        n = index[ci] - 1
        ci -= 1

    return ghat, width, height


def optimal_llr(tar, non, laplace=False, monotonicity_epsilon=1e-6, compute_eer=False):
    """Uses PAV algorithm to optimally calibrate the score

    Parameters
    ----------
    tar : ndarray
        list of scores associated to target pairs
        (uncalibrated LLRs)
    non : ndarray
        list of scores associated to non-target pairs
        (uncalibrated LLRs)
    laplace : bool
        Use Laplace technique to avoid infinite values of LLRs
    monotonicity_epsilon : float
        Unsures monoticity of the optimal LLRs
    compute_eer : bool
        Returns ROCCH-EER

    Returns
    -------
    tar : ndarray
        Target optimally calibrated scores (PAV)
    non : ndarray
        Non-target optimally calibrated scores (PAV)
    eer : float
        ROCCH-EER

    Notes
    -----
    Credits to Andreas Nautsch (EURECOM)
    https://gitlab.eurecom.fr/nautsch/cllr

    References
    ----------

    .. [2] Brümmer, N., & Du Preez, J. (2006).
    Application-independent evaluation of speaker detection.
    Computer Speech & Language, 20(2-3), 230-275.
    """
    scores = np.concatenate([non, tar])
    Pideal = np.concatenate([np.zeros(len(non)), np.ones(len(tar))])

    perturb = np.argsort(scores, kind='mergesort')
    Pideal = Pideal[perturb]

    if laplace:
        Pideal = np.hstack([1, 0, Pideal, 1, 0])

    Popt, width, foo = pavx(Pideal)

    if laplace:
        Popt = Popt[2:len(Popt) - 2]

    posterior_log_odds = logit(Popt)
    log_prior_odds = np.log(len(tar) / len(non))
    llrs = posterior_log_odds - log_prior_odds
    N = len(tar) + len(non)
    llrs = llrs + np.arange(N) * monotonicity_epsilon / N  # preserve monotonicity

    idx_reverse = np.zeros(len(scores), dtype=int)
    idx_reverse[perturb] = np.arange(len(scores))
    tar_llrs = llrs[idx_reverse][len(non):]
    nontar_llrs = llrs[idx_reverse][:len(non)]

    if not compute_eer:
        return tar_llrs, nontar_llrs

    nbins = width.shape[0]
    pmiss = np.zeros(nbins + 1)
    pfa = np.zeros(nbins + 1)
    #
    # threshold leftmost: accept everything, miss nothing
    left = 0  # 0 scores to left of threshold
    fa = non.shape[0]
    miss = 0
    #
    for i in range(nbins):
        pmiss[i] = miss / len(tar)
        pfa[i] = fa /len(non)
        left = int(left + width[i])
        miss = np.sum(Pideal[:left])
        fa = len(tar) + len(non) - left - np.sum(Pideal[left:])
    #
    pmiss[nbins] = miss / len(tar)
    pfa[nbins] = fa / len(non)

    eer = 0
    for i in range(pfa.shape[0] - 1):
        xx = pfa[i:i + 2]
        yy = pmiss[i:i + 2]

        # xx and yy should be sorted:
        assert (xx[1] <= xx[0]) & (yy[0] <= yy[1]), \
            'pmiss and pfa have to be sorted'

        XY = np.column_stack((xx, yy))
        dd = np.dot(np.array([1, -1]), XY)
        if np.min(np.abs(dd)) == 0:
            eerseg = 0
        else:
            # find line coefficients seg s.t. seg'[xx(i);yy(i)] = 1,
            # when xx(i),yy(i) is on the line.
            seg = np.linalg.solve(XY, np.array([[1], [1]]))
            # candidate for EER, eer is highest candidate
            eerseg = 1 / (np.sum(seg))

        eer = max([eer, eerseg])
    return tar_llrs, nontar_llrs, eer



def bayes_error_rate(matedScores, nonMatedScores, prior_log_odds):
    """Comptues the bayes error rate corresponding to the score for different prior log-odds

    prior  P1 = Pr(Hm) (probability of mated or target)
    prior_log_odds = lambda = logit(P1)
    proportion of misses: Pmiss
    proportion of false alarms: Pfa
    returns Pe = P1 Pmiss(-lambda) + (1-P1) Pfa(-lambda)
    Pe is a vector of len(prior_log_odds), i.e., one Pe per prior
    ----------
    matedScores : Array_like
        list of scores associated to mated pairs
    nonMatedScores : Array_like
        list of scores associated to non-mated pairs
    prior_log_odds : Array_like
        list of prior log-odds corresponding to each output


    Returns
    -------
    pe : ndarray
        Bayes error rate per input prior log-odds

    Notes
    -----
    This code was inspired by some function of the BOSARIS
    toolkit by Niko Brümmer and Edward de Villiers
    https://sites.google.com/site/bosaristoolkit/

    References
    ----------

    .. [2] Brümmer, N., & Du Preez, J. (2006).
    Application-independent evaluation of speaker detection.
    Computer Speech & Language, 20(2-3), 230-275.
    """
    pmiss = np.zeros(len(prior_log_odds))
    pfa = np.zeros(len(prior_log_odds))
    pe = np.zeros(len(prior_log_odds))
    for i, v_prior_log_odds in enumerate(prior_log_odds):
        # mated or target trials
        posteriors = np.array([sigmoid(llr + v_prior_log_odds) for llr in matedScores])
        pmiss[i] = np.mean((1 - np.sign(posteriors - 0.5)) / 2)
        # non mated or non target trials
        posteriors = np.array([sigmoid(llr + v_prior_log_odds) for llr in nonMatedScores])
        pfa[i] = np.mean((1 - np.sign(0.5 - posteriors)) / 2)
        # Pe = P1 Pmiss(-lambda) + (1-P1) Pfa(-lambda)
        pe[i] = pmiss[i] * sigmoid(v_prior_log_odds) + pfa[i] * sigmoid(-v_prior_log_odds)
    return pe


def ape_plot(matedScores, nonMatedScores, matedScores_opt, nonMatedScores_opt, cllr, cmin, eer, output_file):
    """Draw both APE-plot for calibrated and uncalibrated input scores

    Parameters
    ----------
    matedScores : Array_like
        list of scores associated to mated pairs
    nonMatedScores : Array_like
        list of scores associated to non-mated pairs
    matedScores_opt : Array_like
        Calibrated mated scores
    nonMatedScores_opt : Array_like
        Calibrated non-mated scores
    cllr : float
        application independent cost-function on uncalibrated scores.
    cmin : float
        application independent cost-function on calibrated scores.
    eer : float
        ROCCH Equal Error Rate.
    output_file : String
        Path to png and pdf output file.

    References
    ----------

    .. [2] Brümmer, N., & Du Preez, J. (2006).
    Application-independent evaluation of speaker detection.
    Computer Speech & Language, 20(2-3), 230-275.
    """
    # Colorblind and photocopy friendly colors
    colors = ['#e66101','#fdb863','#b2abd2','#5e3c99']# ['#edf8b1','#7fcdbb','#2c7fb8']
    legendLocation = 'upper right'
    plt.clf()
    ax = plt.gca()
    # Priors to consider
    #plo = np.concatenate((np.arange(-7, 7, 0.5),np.arange(7,50,2)))
    plo = np.arange(-7, 7, 0.25)
    pe = bayes_error_rate(matedScores, nonMatedScores, plo)
    minPe = bayes_error_rate(matedScores_opt, nonMatedScores_opt, plo)
    refPe = bayes_error_rate([0],[0],plo)
    l3, = plt.plot(plo, refPe, label='$\mathrm{P}^{ref}_{e}$', color='black',linewidth=2, linestyle=':')
    l2, = plt.plot(plo, minPe, label='$\mathrm{P}^{min}_{e}$', color=colors[0],linewidth=2)
    l1, = plt.plot(plo, pe, label='$\mathrm{P}_{e}$', color=colors[3],linewidth=2,linestyle='--')
    leer = plt.plot([min(plo), max(plo)], [eer, eer], label='EER', color='black',linewidth=1,linestyle='-.')
    # Information of the figure
    ax.set_ylabel("P(error)")
    ax.set_xlabel("logit prior")
    ax.set_title("$\mathrm{C}_{LLR}$ = %.2f, $\mathrm{C}_{LLR}^{min}$ = %.2f, EER = %.2f" % (cllr,cmin,eer),  y = 1.02)
    ax.legend(loc = legendLocation)
    # Saving Figure (the replacements of extentions are optional)
    outname= output_file.replace('.pdf', '').replace('.png', '').replace('.csv', '').replace('.txt', '')
    plt.savefig(outname + ".pdf", format="pdf")
    plt.savefig(outname + ".png", format="png")


def writeScores(matedScores, nonMatedScores, output_file):
    """Writes scores in a single file
    One line per score in the form of: "<score_value (float)> <key (1 or 0)>"
    (1 is for mated and 0 is for non-mated)

    Parameters
    ----------
    matedScores : Array_like
        list of scores associated to mated pairs
    nonMatedScores : Array_like
        list of scores associated to non-mated pairs
    output_file : String
        Path to output file.
    """
    keys = np.append(np.zeros(len(nonMatedScores)), np.ones(len(matedScores)))
    scores = np.append(nonMatedScores,matedScores)
    sortedScores = sorted(zip(scores,keys), key=lambda pair: pair[0])
    with open(output_file, 'w') as out_f:
        for i in range(len(sortedScores)):
            score = sortedScores[i][0]
            key = sortedScores[i][1]
            out_f.write("{0} {1}\n".format(score,int(key)))

def readScoresSingleFile(input_file):
    """Read scores from a single file
    One line per score in the form of: "<score_value (float)> <key (1 or 0)>"
    (1 is for mated and 0 is for non-mated)

    Parameters
    ----------
    input_file : String
        Path to the socre file.

    Returns
    -------
    matedScores : Array_like
        list of scores associated to mated pairs
    nonMatedScores : Array_like
        list of scores associated to non-mated pairs
    """
    df = pd.read_csv(input_file, sep=' ', header=None)
    matedScores = df[df[1]==1][0].values
    nonMatedScores = df[df[1]==0][0].values
    return matedScores, nonMatedScores


def my_split(s, seps):
    """Splits a string using multiple separators

    Parameters
    ----------
    s : String
        String to split.
    seps : Array_like
        List of separators
    Returns
    -------
    res : list
        list of tokens from splitting the input string
    """
    res = [s]
    for sep in seps:
        s, res = res, []
        for seq in s:
            res += seq.split(sep)
    return res

def readScoresKaldSpkv(input_file):
    """Read score-file from the kaldi speaker verification protocol

    Parameters
    ----------
    input_file : String
        Path to the socre file.

    Returns
    -------
    matedScores : Array_like
        list of scores associated to mated pairs
    nonMatedScores : Array_like
        list of scores associated to non-mated pairs
    """
    # workarround the kaldi codification of utterance informaiton
    def extract_info_from_scp_key(key):
        tokens = my_split(str(key), '-_')
        if len(tokens) == 7:
            targetId = tokens[3]
            userId = tokens[4]
            chapterId = tokens[5]
            uttId = tokens[6].replace(' ', '')
        elif len(tokens) == 4:
            userId = tokens[0]
            targetId = tokens[0]
            chapterId = tokens[1]
            uttId = tokens[2] + '-' + tokens[3].replace(' ', '')
        elif len(tokens) == 3:
            userId = tokens[0]
            targetId = tokens[0]
            chapterId = tokens[1]
            uttId = tokens[2].replace(' ', '')
        elif len(tokens) == 1:
            userId = tokens[0]
            targetId = tokens[0]
            chapterId = "00000"
            uttId = "0000"
        return userId, "{0}-{1}-{2}".format(targetId,chapterId,uttId)
        #return userId, "{targetId}-{chapterId}-{uttId}".format(targetId,chapterId,uttId)
    df = pd.read_csv(input_file, header=None,dtype={'0':'str','1':'str'}, delimiter=r"\s+", engine='python')
    keys = df.apply(lambda row: extract_info_from_scp_key(row[0])[0] == extract_info_from_scp_key(row[1])[0], axis=1)
    matedScores = df[2].values[keys == True]
    nonMatedScores = df[2].values[keys == False]
    return matedScores, nonMatedScores

def ece(tar, non, plo):
    if isscalar(tar):
        tar = array([tar])
    if isscalar(non):
        non = array([non])
    if isscalar(plo):
        plo = array([plo])

    ece = zeros(plo.shape)
    for i, p in enumerate(plo):
        ece[i] = sigmoid(p) * (-log(sigmoid(tar + p))).mean()
        ece[i] += sigmoid(-p) * (-log(sigmoid(-non - p))).mean()

    ece /= log(2)

    return ece


## OLD Version does not manage well value of x too close to zero.
# def int_ece(x):
#     # see Z(X) function in our paper; here for x as LLRs
#     idx = (~isinf(x)) & (x != 0)
#     contrib = zeros(len(x))
#     contrib[x == infty] = 0.25
#     xx = x[idx]
#     LR = exp(xx)
#     contrib[idx] = (LR**2 - 4*LR + 2*xx + 3) / (4*(LR - 1)**2)
#     LRm1 = exp(xx) - 1
#     contrib[idx] = 0.25 - 1/(2*LRm1) + xx / (2*LRm1**2)
#     return contrib.mean()

def int_ece(x, epsilon=1e-6):
    """
    Z(X) = avg( [(x-3)*(x-1) + 2*log(x)] / [4 * (x-1)^2] )  # x as LR
         = avg( 0.25 - 1/[2*(x-1)] + log(x)/[2*(x-1)^2] )  # a = log(x)
         = 0.25 + 0.5 * avg( - 1/[(exp(a) - 1)] + a/[(exp(a)-1)^2] )  # b = exp(a)-1
         = 0.25 + 0.5 * avg( (a-b))/b^2 )
    """
    idx = (~isinf(x)) & (abs(x) > epsilon)
    contrib = zeros(len(x))  # for +inf, the contribution is 0.25; the later on constant term
    xx = x[idx]
    LRm1 = exp(xx) - 1
    contrib[idx] = (xx - LRm1) / LRm1 ** 2
    # if x == 0 or if x < epsilon
    # numerical issue of exp() function for small values around zero, thus also hardcoded value
    contrib[(abs(x) < epsilon)] = -0.5  # Z(0) = 0 = 0.25 + (-0.5)/2
    return 0.25 + contrib.mean() / 2



def dece(tar_llrs, nontar_llrs):
    int_diff_ece = int_ece(tar_llrs) + int_ece(-nontar_llrs)
    return int_diff_ece / log(2)




def ece_plot(matedScores_opt, nonMatedScores_opt, dece, max_abs_LLR, cat_tag, output_file):
    colors = ['#e66101','#fdb863','#b2abd2','#5e3c99']# ['#edf8b1','#7fcdbb','#2c7fb8']
    figureTitle = ''
    if figureTitle == '':
     figureTitle = 'Clean'
    legendLocation = 'upper right'
    plt.clf()
    ax = plt.gca()
    # Prior to consider
    #plo = np.concatenate((np.arange(-7, 7, 0.5),np.arange(7,50,2)))
    plo = np.arange(-7, 7, 0.25)
    minPe = ece(matedScores_opt, nonMatedScores_opt, plo)
    refPe = ece(np.array([0]),np.array([0]),plo)
    # or for ref self.defECE = (sigmoid(self.plo) * -log(sigmoid(self.plo)) + sigmoid(-self.plo) * -log(sigmoid(-self.plo))) / log(2)
    l3, = plt.plot(plo, refPe, label='$\mathrm{ECE}^{ref}$', color='black',linewidth=2, linestyle=':')
    l2, = plt.plot(plo, minPe, label='$\mathrm{ECE}$', color=colors[0],linewidth=2)
    #l1, = plt.plot(plo, pe, label='$\mathrm{P}_{e}$', color=colors[3],linewidth=2,linestyle='--')
    #leer = plt.plot([min(plo), max(plo)], [eer, eer], label='EER', color='black',linewidth=1,linestyle='-.')
    # Information of the figure
    ax.set_ylabel("ECE (bits)")
    ax.set_xlabel("logit prior")
    ax.set_title("$\mathrm{D}_{\mathrm{ECE}}$ = %.2f, $max_{|llr|}$ = %.2f, %s" % (dece,max_abs_LLR,cat_tag),  y = 1.02)
    ax.legend(loc = legendLocation)
    # Saving Figure (the replacements of extentions are optional)
    outname= output_file.replace('.pdf', '').replace('.png', '').replace('.csv', '').replace('.txt', '')
    plt.savefig(outname + ".pdf", format="pdf")
    plt.savefig(outname + ".png", format="png")



def max_abs_LLR(matedScores_opt, nonMatedScores_opt):
	max_abs_LLR = abs(hstack((matedScores_opt,nonMatedScores_opt))).max() / log(10)
	return max_abs_LLR


def category_tag_evidence(max_abs_LLR):
	# smallest float value we can numerically tract in this computational environment
	eps = finfo(float).eps

	# Here are our categorical tags, inspired by the ENFSI sacle on the stength of evidence
	# Please feel free to try out your own scale as well :)
	# dict: { TAG : [min max] value of base10 LLRs }
	categorical_tags = {
	    '0': array([0, eps]),
	    'A': array([eps, 1]),
	    'B': array([1, 2]),
	    'C': array([2, 4]),
	    'D': array([4, 5]),
	    'E': array([5, 6]),
	    'F': array([6, inf])
	}

	# pre-computation for easier later use
	cat_ranges = vstack(list(categorical_tags.values()))
	cat_idx = argwhere((cat_ranges < max_abs_LLR).sum(1) == 1).squeeze()
	cat_tag = list(categorical_tags.keys())[cat_idx]
	return cat_tag
