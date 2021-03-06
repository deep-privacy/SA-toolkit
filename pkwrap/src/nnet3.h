#ifndef PKWRAP_NNET3_H_
#define PKWRAP_NNET3_H_
#include "common.h"
#include "matrix.h"
#include <torch/extension.h>
#include "nnet3/natural-gradient-online.h"
kaldi::BaseFloat precondition_directions(kaldi::nnet3::OnlineNaturalGradient &state, torch::Tensor &grad);
#endif
