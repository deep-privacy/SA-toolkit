import torch
import torch.nn as nn

import random
import pkwrap
from pkwrap.hifigan import F0QuantModel

import configargparse
import sys

import logging

logging.basicConfig(level=logging.INFO)


def build(args):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = pkwrap.nn.Encoder(
                input_emb_width=1,
                output_emb_width=128,
                levels=1,
                downs_t=[4],
                strides_t=[2],
                width=32,
                depth=4,
                m_conv=1.0,
                dilation_growth_rate=3,
            )
            self.vq = pkwrap.nn.Bottleneck(l_bins=20, emb_width=128, mu=0.99, levels=1)
            self.decoder = pkwrap.nn.Decoder(
                input_emb_width=1,
                output_emb_width=128,
                levels=1,
                downs_t=[4],
                strides_t=[2],
                width=32,
                depth=4,
                m_conv=1.0,
                dilation_growth_rate=3,
            )

        @torch.no_grad()
        def validate_model(self, device="cpu"):
            N = 2
            C = 10 * 2740
            x = torch.arange(N * C).reshape(N, 1, C).float().to(device)
            nnet_output, _, _ = self.forward(f0=x)
            assert nnet_output.shape == (
                2,
                1,
                27392,
            ), f"{nnet_output.shape} != expected frame subsampling"

            self.eval()
            nnet_output, _, _ = self.forward(f0=x)
            assert nnet_output.shape == (
                2,
                1,
                27392,
            ), f"{nnet_output.shape} != expected frame subsampling"
            self.train()

        def forward(self, **kwargs):
            f0_h = self.encoder(kwargs["f0"])
            _, f0_h_q, f0_commit_losses, f0_metrics = self.vq(f0_h)
            f0 = self.decoder(f0_h_q)

            return f0, f0_commit_losses, f0_metrics

    return Net


if __name__ == "__main__":
    parser = configargparse.ArgumentParser(description="Model config args")
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--checkpoint_path", default="exp/f0_vq", type=str)
    parser.add_argument("--training_epochs", default=100, type=int)
    args, remaining_argv = parser.parse_known_args()
    sys.argv = sys.argv[:1] + remaining_argv

    torch.cuda.manual_seed(52)
    random.seed(52)

    wav_list = pkwrap.utils.fs.scans_directory_for_ext(
        "./data/LJSpeech-1.1/wavs_16khz", "wav"
    )
    wav_list.sort()
    random.shuffle(wav_list)
    split = int(0.92 * len(wav_list))
    train_list = ",".join(wav_list[:split])
    dev_list = ",".join(wav_list[split:])

    F0QuantModel(
        build(args),
        **{
            "mode": "train",
            "training_epochs": args.training_epochs,
            "num_workers": 4,
            "rank": args.local_rank,
            "checkpoint_path": args.checkpoint_path,
            "init_weight_model": "last",
            "train_utterances": train_list,
            "test_utterances": dev_list,
        },
    )
