import torchaudio
from torchaudio.models.wav2vec2.utils.import_fairseq import _convert_state_dict as _convert
from torchaudio.models.wav2vec2.model import Wav2Vec2Model


import torch

# In part from by:
# https://github.com/pytorch/audio/blob/fee994ce755d29ec62806df19e5359c7f8851605/tools/convert_voxpopuli_models.py


def _removeprefix(s, prefix):
    if s.startswith(prefix):
        return s[len(prefix) :]
    return s


def _load(input_file):
    if input_file.startswith("http"):
        data = torch.hub.load_state_dict_from_url(input_file)
    else:
        data = torch.load(input_file)

    #  print(data["cfg"])
    cfg = data["cfg"]
    for key in list(cfg.keys()):
        if key != "model":
            del cfg[key]
            if "w2v_args" in cfg["model"]:
                del cfg["model"]["w2v_args"][key]
    state_dict = {_removeprefix(k, "w2v_encoder."): v for k, v in data["model"].items()}
    return cfg, state_dict


def _parse_model_param(cfg, state_dict):
    """
        To get the model param from fairseq model
        cfg, state_dict = _load(input_file)
        params = _parse_model_param(cfg, state_dict)
    """
    key_mapping = {
        "extractor_mode": "extractor_mode",
        "conv_feature_layers": "extractor_conv_layer_config",
        "conv_bias": "extractor_conv_bias",
        "encoder_embed_dim": "encoder_embed_dim",
        "dropout_input": "encoder_projection_dropout",
        "conv_pos": "encoder_pos_conv_kernel",
        "conv_pos_groups": "encoder_pos_conv_groups",
        "encoder_layers": "encoder_num_layers",
        "encoder_attention_heads": "encoder_num_heads",
        "attention_dropout": "encoder_attention_dropout",
        "encoder_ffn_embed_dim": "encoder_ff_interm_features",
        "activation_dropout": "encoder_ff_interm_dropout",
        "dropout": "encoder_dropout",
        "layer_norm_first": "encoder_layer_norm_first",
        "layerdrop": "encoder_layer_drop",
        "encoder_layerdrop": "encoder_layer_drop",
    }
    params = {}
    src_dicts = [cfg["model"]]
    if "w2v_args" in cfg["model"]:
        src_dicts.append(cfg["model"]["w2v_args"]["model"])

    for src, tgt in key_mapping.items():
        for model_cfg in src_dicts:
            if src in model_cfg:
                params[tgt] = model_cfg[src]
                break
    if params["extractor_mode"] == "default":
        params["extractor_mode"] = "group_norm"
    # the following line is commented out to resolve lint warning; uncomment before running script
    params["extractor_conv_layer_config"] = eval(params["extractor_conv_layer_config"])
    assert len(params) == 15
    params["aux_num_out"] = state_dict["proj.bias"].numel() if "proj.bias" in state_dict else None
    return params



def wav2vec2_model(
    *args, **kwargs
) -> Wav2Vec2Model:
    """
    from satools.utils.import_fairseq_model import wav2vec2_model
    m = wav2vec2_model(**{
        "extractor_mode": "layer_norm",
        "extractor_conv_layer_config": [[ 512, 10, 5 ], [ 512, 3, 2 ], [ 512, 3, 2 ], [ 512, 3, 2 ], [ 512, 3, 2 ], [ 512, 2, 2 ], [ 512, 2, 2 ]],
        "extractor_conv_bias": True,
        "encoder_embed_dim": 1024,
        "encoder_projection_dropout": 0.0,
        "encoder_pos_conv_kernel": 128,
        "encoder_pos_conv_groups": 16,
        "encoder_num_layers": 24,
        "encoder_num_heads": 16,
        "encoder_attention_dropout": 0.0,
        "encoder_ff_interm_features": 4096,
        "encoder_ff_interm_dropout": 0.0,
        "encoder_dropout": 0.0,
        "encoder_layer_norm_first": True,
        "encoder_layer_drop": 0.0,
        "aux_num_out": None
    })
    m.load_convert_checkpoint("/lium/home/pchampi/.cache/torch/hub/checkpoints/wav2vec2_large_west_germanic_v2.pt")
    """

    def fairseq_load_conert(self_class, file, strict=False):
        _, state_dict = _load(file)
        return self_class.load_state_dict(**{"state_dict": _convert(state_dict), "strict": strict})
    Wav2Vec2Model.load_convert_checkpoint = fairseq_load_conert

    _model = torchaudio.models.wav2vec2.model.wav2vec2_model(*args, **kwargs)
    return _model
