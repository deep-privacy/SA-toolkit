record_audio_colab = """
  async function recordAudio() {
    const div = document.createElement('div');
    const audio = document.createElement('audio');
    const strtButton = document.createElement('button');
    const stopButton = document.createElement('button');
    strtButton.textContent = 'Start Recording';
    stopButton.textContent = 'Stop Recording';
    
    document.body.appendChild(div);
    div.appendChild(strtButton);
    div.appendChild(audio);
    const stream = await navigator.mediaDevices.getUserMedia({audio:true});
    let recorder = new MediaRecorder(stream);
    
    audio.style.display = 'block';
    audio.srcObject = stream;
    audio.controls = true;
    audio.muted = true;
    await new Promise((resolve) => strtButton.onclick = resolve);
      strtButton.replaceWith(stopButton);
      recorder.start();
    await new Promise((resolve) => stopButton.onclick = resolve);
      recorder.stop();
      let recData = await new Promise((resolve) => recorder.ondataavailable = resolve);
      let arrBuff = await recData.data.arrayBuffer();
      stream.getAudioTracks()[0].stop();
      div.remove()
      let binaryString = '';
      let bytes = new Uint8Array(arrBuff);
      bytes.forEach((byte) => { binaryString += String.fromCharCode(byte) });
    const url = URL.createObjectURL(recData.data);
    const player = document.createElement('audio');
    player.controls = true;
    player.src = url;
    document.body.appendChild(player);
  return btoa(binaryString)};
"""

"""
For IPyhon dislpay of WER:
"""


def ComputeEditDistanceMatrix(hs, rs):
    """Compute edit distance between two list of strings.
    Args:
      hs: the list of words in the hypothesis sentence
      rs: the list of words in the reference sentence
    Returns:
      Edit distance matrix (in the format of list of lists), where the first
      index is the reference and the second index is the hypothesis.
    """
    dr, dh = len(rs) + 1, len(hs) + 1
    dists = [[]] * dr

    # Initialization.
    for i in range(dr):
        dists[i] = [0] * dh
        for j in range(dh):
            if i == 0:
                dists[0][j] = j
            elif j == 0:
                dists[i][0] = i

    # Do dynamic programming.
    for i in range(1, dr):
        for j in range(1, dh):
            if rs[i - 1] == hs[j - 1]:
                dists[i][j] = dists[i - 1][j - 1]
            else:
                tmp0 = dists[i - 1][j - 1] + 1
                tmp1 = dists[i][j - 1] + 1
                tmp2 = dists[i - 1][j] + 1
                dists[i][j] = min(tmp0, tmp1, tmp2)

    return dists


def _GenerateAlignedHtml(hyp, ref, err_type):
    """Generate a html element to highlight the difference between hyp and ref.
    Args:
      hyp: Hypothesis string.
      ref: Reference string.
      err_type: one of 'none', 'sub', 'del', 'ins'.
    Returns:
      a html string where disagreements are highlighted.
        - hyp highlighted in red, and marked with <del> </del>
        - ref highlighted in green
    """

    highlighted_html = ""
    if err_type == "none":
        highlighted_html += "%s " % hyp

    elif err_type == "sub":
        highlighted_html += """<span style="background-color: #a25239">
        <del>%s</del></span><span style="background-color: #63ae5d">
        %s </span> """ % (
            hyp,
            ref,
        )

    elif err_type == "del":
        highlighted_html += """<span style="background-color: #63ae5d">
        %s</span> """ % (
            ref
        )

    elif err_type == "ins":
        highlighted_html += """<span style="background-color: #a25239">
        <del>%s</del> </span> """ % (
            hyp
        )

    else:
        raise ValueError("unknown err_type " + err_type)

    return highlighted_html


def GenerateSummaryFromErrs(nref, errs):
    """Generate strings to summarize word errors.
    Args:
      nref: integer of total words in references
      errs: dict of three types of errors. e.g. {'sub':10, 'ins': 15, 'del': 3}
    Returns:
      str1: string summarizing total error, total word, WER,
      str2: string breaking down three errors: deleting, insertion, substitute
    """

    total_error = sum(errs.values())
    str_sum = "total error = %d, total word = %d, wer = %.2f%%" % (
        total_error,
        nref,
        total_error * 100.0 / nref,
    )

    str_details = "Error breakdown: del = %.2f%%, ins=%.2f%%, sub=%.2f%%" % (
        errs["del"] * 100.0 / nref,
        errs["ins"] * 100.0 / nref,
        errs["sub"] * 100.0 / nref,
    )

    return str_sum, str_details


def computeWER(hyp, ref, diagnosis=True):
    """Computes WER for ASR by ignoring diff of punctuation, space, captions.
    Args:
      hyp: Hypothesis string.
      ref: Reference string.
      diagnosis (optional): whether to generate diagnosis str (in html format)
    Returns:
      dict of three types of errors. e.g. {'sub':0, 'ins': 0, 'del': 0}
      num of reference words, integer
      aligned html string for diagnois (empty if diagnosis = False)
    """

    # Compute edit distance.
    hs = hyp.split()
    rs = ref.split()
    distmat = ComputeEditDistanceMatrix(hs, rs)

    # Back trace, to distinguish different errors: insert, deletion, substitution.
    ih, ir = len(hs), len(rs)
    errs = {"sub": 0, "ins": 0, "del": 0}
    aligned_html = ""
    while ih > 0 or ir > 0:
        err_type = ""

        # Distinguish error type by back tracking
        if ir == 0:
            err_type = "ins"
        elif ih == 0:
            err_type = "del"
        else:
            if hs[ih - 1] == rs[ir - 1]:  # correct
                err_type = "none"
            elif distmat[ir][ih] == distmat[ir - 1][ih - 1] + 1:  # substitute
                err_type = "sub"
            elif distmat[ir][ih] == distmat[ir - 1][ih] + 1:  # deletion
                err_type = "del"
            elif distmat[ir][ih] == distmat[ir][ih - 1] + 1:  # insert
                err_type = "ins"
            else:
                raise ValueError("fail to parse edit distance matrix")

        # Generate aligned_html
        if diagnosis:
            if ih == 0 or not hs:
                tmph = " "
            else:
                tmph = hs[ih - 1]
            if ir == 0 or not rs:
                tmpr = " "
            else:
                tmpr = rs[ir - 1]
            aligned_html = _GenerateAlignedHtml(tmph, tmpr, err_type) + aligned_html

        # If no error, go to previous ref and hyp.
        if err_type == "none":
            ih, ir = ih - 1, ir - 1
            continue

        # Update error.
        errs[err_type] += 1

        # Adjust position of ref and hyp.
        if err_type == "del":
            ir = ir - 1
        elif err_type == "ins":
            ih = ih - 1
        else:  # err_type == 'sub'
            ih, ir = ih - 1, ir - 1

    assert distmat[-1][-1] == sum(errs.values())

    # Num of words. For empty ref we set num = 1.
    nref = max(len(rs), 1)

    if aligned_html:
        str1, str2 = GenerateSummaryFromErrs(nref, errs)
        aligned_html = str1 + " (" + str2 + ")" + "<br>" + aligned_html

    return errs, nref, aligned_html
