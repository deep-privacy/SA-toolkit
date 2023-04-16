#ifndef PKWRAP_DECODER_H_
#define PKWRAP_DECODER_H_
#include "common.h"
#include "fstext/fstext-lib.h"
#include "decoder/lattice-faster-decoder.h"
#include "decoder/decoder-wrappers.h"
#include "decoder/decodable-matrix.h"
#include "decoder/faster-decoder.h"
#include "lat/lattice-functions.h"
#include <tuple>

kaldi::LatticeFasterDecoderConfig CreateLatticeFasterDecoderConfig(float beam, int32 max_active, int32 min_active, float lattice_beam);

std::tuple<std::string, std::vector<int32>, std::vector<int32>> MappedLatticeFasterRecognizer(
        kaldi::Matrix<kaldi::BaseFloat> loglikes,
        kaldi::LatticeFasterDecoderConfig &config,
        std::string &trans_model_s,
        std::string &fst_in_str,
        std::string &word_syms_filename,
        float acoustic_scale,
        bool allow_partial
        );

#endif

