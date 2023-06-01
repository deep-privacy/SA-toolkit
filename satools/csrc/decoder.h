#ifndef PKWRAP_DECODER_H_
#define PKWRAP_DECODER_H_
#include "common.h"
#include "fstext/fstext-lib.h"
#include "decoder/lattice-faster-decoder.h"
#include "decoder/decoder-wrappers.h"
#include "decoder/decodable-matrix.h"
#include "decoder/faster-decoder.h"
#include "lat/lattice-functions.h"
#include "lm/const-arpa-lm.h"
#include "lat/word-align-lattice-lexicon.h"
#include <tuple>

kaldi::LatticeFasterDecoderConfig CreateLatticeFasterDecoderConfig(float beam, int32 max_active, int32 min_active, float lattice_beam);

std::tuple<std::string, std::vector<int32>, std::vector<int32>, kaldi::CompactLattice> MappedLatticeFasterRecognizer(
        kaldi::Matrix<kaldi::BaseFloat> loglikes,
        kaldi::LatticeFasterDecoderConfig &config,
        std::string &trans_model_s,
        std::string &fst_in_str,
        std::string &word_syms_filename,
        float acoustic_scale,
        bool allow_partial
        );

kaldi::CompactLattice LatticeLmrescore(
        fst::VectorFst<fst::StdArc>  &std_lm_fst_in,
        kaldi::CompactLattice &clat,
        float lm_scale_
        );

kaldi::CompactLattice LatticeLmrescoreConstArpa(
        std::string  &constarpalm_filename,
        kaldi::CompactLattice &clat,
        float lm_scale_
        );

std::tuple<std::string, std::vector<int32>, std::vector<int32>, kaldi::CompactLattice> LatticeBestPath(
        std::string &trans_model_s,
        std::string &word_syms_filename,
        kaldi::CompactLattice &clat,
        float acoustic_scale_,
        float lm_scale_
        );

kaldi::CompactLattice LatticeAlignWordsLexicon(
        std::string &trans_model_s,
        std::string &align_lexicon_rxfilename,
        kaldi::CompactLattice &clat
        );

std::tuple<std::vector<int32>, std::vector<float>,std::vector<float>, std::vector<std::string>> NbestToCTM(
        std::string &word_syms_filename,
        kaldi::CompactLattice &clat,
        float frame_shift_,
        bool print_silence
        );

#endif

