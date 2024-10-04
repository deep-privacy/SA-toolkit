import logging

import satools.infer_helper as infer_helper
import torch
import torch.nn as nn
import torchaudio

import satools

logging.basicConfig(level=logging.INFO)

def build():
    class Net(nn.Module):
        def __init__(self, speaker_number, loss=None, embedding_size=256):
            super().__init__()

            self.bn_extractor_func, self.model = satools.infer_helper.init_asr_model(
                model="local/chain/e2e/tuning/tdnnf_spkadv.py", exp_path="exp/chain/e2e_tdnnf/", additional_args={
                    "adversarial_training": "False",
                    "spk2id": "./data/spk2id",
                })
            self.print_text = False

            time_masking = torchaudio.transforms.TimeMasking(time_mask_param=5)
            freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=10)

            def spec_augment(x):
                out = x.permute(0, 2, 1)
                out = freq_masking(out)
                out = time_masking(out)
                out = out.permute(0, 2, 1)
                return out

            self.spec_aug = spec_augment

        def set_lr_weight_decay_layers_for_optim(self, _optimizer, _options):

            optimizer, _ = self.model.asi.get_optimizer(
                iter=0, learning_rate=_options["lr"]
            )

            return optimizer

        def new_epoch_hook(self, current_epoch, total_epoch):
            self.print_text = True

        def forward(self, args, target=None, norm_embedding=True):
            """
            The forward mothod MUST take 3 arguemnts
            The forward mothod MUST return 2 values:
               - a tuple of: (loss: to train the model, in testing (target==None) you should return torch.tensor(torch.nan).
                               cross-entroy prediction: raw output of the network to compute accuracy on
                 - In this example the returned value handled by: ArcMarginProduct
               - the x-vector embedding
               i.e., (loss, cce), x_vector = model([...])
            """

            aug = lambda x: x
            if self.training:
                aug = self.spec_aug

            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=False):
                    out_asr, model = self.bn_extractor_func(args, aug)
                    out = model.bottleneck_out.permute(0, 2, 1).contiguous()

                if self.print_text:
                    self.print_text = False
                    text = infer_helper.kaldi_asr_decode(out_asr)  # is this even text ?
                    print("Text:", text)

            return self.model.asi(out, target)

        @torch.no_grad()
        def validate_model(self):

            batch = torch.rand(16, 32000).to("cuda")

            indices = torch.randint(0, 5, size=(16,))

            _, x_vector = self.forward(batch)
            assert x_vector.shape[1] == 256

    return Net
