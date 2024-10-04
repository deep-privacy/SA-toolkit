import sys

import torch

import satools

#  python3 ./shutil/diff_models_weights.py ./exp/chain/e2e_tdnnf_vq_sizeco_64/final.pt ./exp/chain/e2e_tdnnf_vq_spkdelta_sizeco_64/final.pt

diffa = torch.load(sys.argv[1], weights_only=False)

diffb = torch.load(sys.argv[2], weights_only=False)

diffb_same_keys_as_a, unmatch = satools.utils.torch.match_state_dict(
    diffa,
    diffb,
)

for a in diffa.items():
    if a[0] in unmatch.keys():
        print("Not Compatible weight:", unmatch.keys())
        continue

    if ".bn." in a[0]: # ignore batch norm
        continue
    if a[1].ndim >=2:
        print(torch.allclose(a[1], diffb_same_keys_as_a[a[0]], atol=1e-12), "\t -- ", a[0], "\t--", (a[1] - diffb_same_keys_as_a[a[0]]).sum(), "\t", a[1][0][:3], diffb_same_keys_as_a[a[0]][0][:3])
        continue
    print(torch.allclose(a[1], diffb_same_keys_as_a[a[0]], atol=1e-12), "\t -- ", a[0], "\t--", (a[1] - diffb_same_keys_as_a[a[0]]).sum(), "\t", a[1][:3], diffb_same_keys_as_a[a[0]][:3])

