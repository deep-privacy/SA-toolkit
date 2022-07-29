#!/usr/bin/env bash

# Copyright 2014-2021 Pierre Champion

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
    echo "Please provide the models path."
    echo "   release_model.sh model_wavlm_ecapa_circle/best_model_wavlm_cuda_JIT.pt model_wavlm_ecapa_circle/best_model_wavlm_cpu_JIT.pt model_wavlm_ecapa_circle/best_model_wavlm.pt"
    exit 1
fi

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "(${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "After creating an account https://huggingface.co/join"
log "Create a new repo here: https://huggingface.co/new"
echo  "                           with a name like {model_name}_{dataset_name}_{epoch}..."

echo ""
read -r -p "Enter repo path: (i.e. daniel/{model_name}_{dataset_name}_{epoch}...): " hf_repo

dir_repo="hugging_face_upload_$hf_repo"
dir_repo=${dir_repo//\//.}

[ ! -d "${dir_repo}" ] && git clone https://huggingface.co/${hf_repo} ${dir_repo}

cd ${dir_repo}

gitlfs=$(git lfs --version 2> /dev/null || true)
        [ -z "${gitlfs}" ] && \
            log "ERROR: You need to install git-lfs first ($ cd ${dir_repo} && git lfs install)" && \
            exit 1
git lfs install
cd -

log "Copying file to ${dir_repo}"
cp $@ ${dir_repo}

cd ${dir_repo}

cat > README.md <<EOF
---
tags:
- pchampio
- audio
inference: false
---
EOF
if [ -n "$(git status --porcelain)" ]; then
    git add .
    git commit -m "Update model"
fi
git push

cd ..
git lfs uninstall
