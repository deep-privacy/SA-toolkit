import fairseq
import torch

feat_model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
    [str("./wav2vec2_base_en_v2.pt")]
)
print(feat_model)
print(cfg)
print(task)
feat_model = feat_model[0]

N = 1
C = 16002
x = torch.arange(N * C).reshape(N, C).float()

print(x.shape)
bottleneck_out = feat_model(x, mask=False, features_only=True)["x"]

print(bottleneck_out.shape, C / bottleneck_out.shape[1])

# ssl model subsample=320, but loss 1 dim that need to add back
bottleneck_out = bottleneck_out.transpose(2, 1)
print(bottleneck_out[0][-2:][:10])
bottleneck_out = torch.nn.functional.pad(bottleneck_out, (0, 1), "replicate")
print(bottleneck_out[0][-2:][:10])
bottleneck_out = bottleneck_out.transpose(2, 1)

print(bottleneck_out.shape, C / bottleneck_out.shape[1])
