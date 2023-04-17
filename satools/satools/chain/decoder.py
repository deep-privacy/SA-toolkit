import torch

"""
This file contains the python side to the kaldi (pkwrap-main.h) bindings for decoding/rescoding/get_word_alignment.
These are nice for few-utterances decoding, but for more, use kaldi with shutil/decode/latgen-faster-mapped.sh
(In here, for each call, trans_model, HCLG, words_txt, ... are reloaded (which is inefficient))
"""

def kaldi_decode(loglikes,
                 trans_model,
                 HCLG,
                 words_txt,
                 opts={
                     "beam":15.0,
                     "max_active":7000,
                     "min_active":200,
                     "lattice_beam":8.0, # Beam we use in lattice generation.
                 },
                 acoustic_scale=1.0,
                 allow_partial=True
                 ):
    """
    Decode loglikes from a tensor, no lm rescoding is done

    Example:
        import torch
        import torchaudio
        import satools

        net = satools.infer_helper.load_model("/lium/scratch/pchampi/SA/egs/asr/librispeech/exp/chain/asr_eval_tdnnf_360h/final.pt")
        wav, _ = torchaudio.load("/lium/scratch/pchampi/SA/egs/anon/vctk/data/vctk_test/wav/p225/p225_001_mic2.wav")
        net = net.cuda()
        loglike, _ = net(wav.cuda())
        loglike = loglike.squeeze(0).cpu()

        print(wav.shape, loglike.shape)

        txt, words, alignment, latt = satools.chain.decoder.kaldi_decode(loglike,
                                              trans_model="/lium/scratch/pchampi/SA/egs/asr/librispeech/exp/chain/asr_eval_tdnnf_360h/0.trans_mdl",
                                              HCLG="/lium/scratch/pchampi/SA/egs/asr/librispeech/exp/chain/e2e_train_clean_360/e2e_biphone_tree/graph_tgsmall/HCLG.fst",
                                              words_txt="/lium/scratch/pchampi/SA/egs/asr/librispeech/exp/chain/e2e_train_clean_360/e2e_biphone_tree/graph_tgsmall/words.txt",
                                              )

    """

    from _satools import kaldi

    conf = kaldi.decoder.CreateLatticeFasterDecoderConfig(opts["beam"], opts["max_active"], opts["min_active"], opts["lattice_beam"])

    s = kaldi.decoder.MappedLatticeFasterRecognizer(
        kaldi.matrix.TensorToKaldiMatrix(loglikes),
        conf,
        trans_model,
        HCLG,
        words_txt,
        acoustic_scale,
        allow_partial,
    )
    return s

def kaldi_lm_rescoring(lat,
                       trans_model,
                       G_old,
                       G_new,
                       words_txt,
                       acoustic_scale=1.0,
                       lm_scale=1.0,
                       ):
    """
    Kaldi Lm rescoring

    Example:
        txt, words, alignment, latt_res = satools.chain.decoder.kaldi_lm_rescoring(latt,
                                                                                trans_model = "/lium/scratch/pchampi/SA/egs/asr/librispeech/exp/chain/asr_eval_tdnnf_360h/0.trans_mdl",
                                                                                G_old = "/lium/scratch/pchampi/SA/egs/asr/librispeech/data/lang_lp_test_tgsmall/G.fst",
                                                                                G_new = "/lium/scratch/pchampi/SA/egs/asr/librispeech/data/lang_lp_test_fglarge/G.carpa",
                                                                                words_txt = "/lium/scratch/pchampi/SA/egs/asr/librispeech/exp/chain/e2e_train_clean_360/e2e_biphone_tree/graph_tgsmall/words.txt")
    """

    from _satools import kaldi

    fst = kaldi.fst.StdVectorFst()
    kaldi.fst.ReadFstKaldi(G_old, fst)
    project_output = True
    kaldi.fst.Project(fst, project_output)

    _acoustic_scale = -1.0
    lm_rescore_lat = kaldi.decoder.LatticeLmrescore(fst, lat, _acoustic_scale)
    _acoustic_scale = 1.0
    const_arpa_lm_rescore_lat = kaldi.decoder.LatticeLmrescoreConstArpa(G_new, lm_rescore_lat, _acoustic_scale)

    txt, words, alignment, lat = kaldi.decoder.LatticeBestPath(trans_model, words_txt, const_arpa_lm_rescore_lat, acoustic_scale, lm_scale)
    return txt, words, alignment, const_arpa_lm_rescore_lat # return the LatticeLmrescoreConstArpa lattice!


def kaldi_get_align(lat,
                    trans_model,
                    align_lexicon,
                    words_txt,
                    frame_shift=0.030,
                    acoustic_scale=1.0,
                    lm_scale=1.0,
                    ):
    """
    Example:
        align = satools.chain.decoder.kaldi_get_align(latt_res,
                                                 trans_model = "/lium/scratch/pchampi/SA/egs/asr/librispeech/exp/chain/asr_eval_tdnnf_360h/0.trans_mdl",
                                                 align_lexicon = "/lium/scratch/pchampi/SA/egs/asr/librispeech/data/lang_lp/phones/align_lexicon.int",
                                                 words_txt = "/lium/scratch/pchampi/SA/egs/asr/librispeech/exp/chain/e2e_train_clean_360/e2e_biphone_tree/graph_tgsmall/words.txt",
                                                 )
    Then use mpv to play the word:
        mpv XX.wav  --start=4.320000 --length=0.270000
    """

    from _satools import kaldi

    ali_lat = kaldi.decoder.LatticeAlignWordsLexicon(trans_model, align_lexicon, lat)

    txt, words, alignment, best_lat = kaldi.decoder.LatticeBestPath(trans_model, words_txt, ali_lat, acoustic_scale, lm_scale)
    print_slience = False
    ctm = list(zip(*kaldi.decoder.NbestToCTM(words_txt, best_lat, frame_shift, print_slience)))
    return ctm

