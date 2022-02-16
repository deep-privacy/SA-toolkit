#!/bin/bash

set -e

nj=$(nproc)

home=$PWD

# CUDA version
CUDAROOT=/usr/local/cuda
if [ "$(id -g --name)" == "lium" ]; then
  CUDAROOT=/opt/cuda/10.2 # LIUM Cluster
  echo "Using local \$CUDAROOT: $CUDAROOT"
fi
cuda_version=$($CUDAROOT/bin/nvcc --version | grep "Cuda compilation tools" | cut -d" " -f5 | sed s/,//)
cuda_version_witout_dot=$(echo $cuda_version | xargs | sed 's/\.//')

# CONDA
conda_url=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
conda_url=https://repo.anaconda.com/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh

# PYTORCH
torch_version=1.8.2
torchvision_version=0.9.2
torchaudio_version=0.8.2

torch_wheels="https://download.pytorch.org/whl/lts/1.8/torch_lts.html"

venv_dir=$PWD/venv

mark=.done-venv
if [ ! -f $mark ]; then
  echo " == Making python virtual environment =="
  name=$(basename $conda_url)
  if [ ! -f $name ]; then
    wget $conda_url || exit 1
  fi
  [ ! -f $name ] && echo "File $name does not exist" && exit 1
  [ -d $venv_dir ] && rm -r $venv_dir
  sh $name -b -p $venv_dir || exit 1
  . $venv_dir/bin/activate

  echo "Installing conda dependencies"
  yes | conda install -c conda-forge sox
  yes | conda install -c conda-forge libflac
  touch $mark
fi
source $venv_dir/bin/activate

export PATH=$CUDAROOT/bin:$PATH
export LD_LIBRARY_PATH=$CUDAROOT/lib64:$LD_LIBRARY_PATH
export CFLAGS="-I$CUDAROOT/include $CFLAGS"
export CUDA_HOME=$CUDAROOT
export CUDA_PATH=$CUDAROOT

export OPENFST_PATH=$(realpath .)/kaldi/tools/openfst
export LD_LIBRARY_PATH=$OPENFST_PATH/lib:$LD_LIBRARY_PATH

echo "if [ \$(which python) != $venv_dir/bin/python ]; then source $venv_dir/bin/activate; fi; export CUDAROOT=$CUDAROOT; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH;" > env.sh

export CPPFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
export CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"

mark=.done-python-requirements-kaldi-feat
if [ ! -f $mark ]; then
  yes | conda install -c conda-forge cmake
  yes | conda install -c conda-forge cudnn

  export CUDNN_ROOT="$venv_dir"
  export CUDNN_INCLUDE_DIR="$venv_dir/include"
  export CUDNN_LIBRARY="$venv_dir/lib/libcudnn.so"
  export KALDIFEAT_CMAKE_ARGS="-DCUDNN_LIBRARY=$CUDNN_LIBRARY -DCMAKE_BUILD_TYPE=Release"
  export KALDIFEAT_MAKE_ARGS="-j"

  pip3 install kaldifeat==1.12
  cd $home
  python3 -c "import kaldifeat; print('Kaldifeat version:', kaldifeat.__version__)" || exit 1
  touch $mark
fi

# mark=.done-k2
# if [ ! -f $mark ]; then
  # echo " == Installing k2 =="
  # export CUDNN_ROOT="$venv_dir"
  # export CUDNN_INCLUDE_DIR="$venv_dir/include"
  # export CUDNN_LIBRARY="$venv_dir/lib/libcudnn.so"
  # git clone https://github.com/k2-fsa/k2.git
  # cd k2
  # python3 setup.py install
  # cd $home
  # touch $mark
# fi


mark=.done-pytorch
if [ ! -f $mark ]; then
  echo " == Installing pytorch $torch_version for cuda $cuda_version =="
  # pip3 install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
  pip3 install torch==$torch_version+cu$cuda_version_witout_dot torchvision==$torchvision_version+cu$cuda_version_witout_dot torchaudio==$torchaudio_version -f $torch_wheels
  cd $home
  touch $mark
fi

mark=.done-python-requirements
if [ ! -f $mark ]; then
  echo " == Installing python libraries =="

  pip3 install numpy==1.20 # force numpy version to 1.20 (required by Numba)

  pip3 install scikit-learn==0.24.2
  pip3 install tensorboard
  pip3 install carbontracker==1.1.6

  # pkwrap additional req
  pip3 install pytorch-memlab==0.2.3
  pip3 install kaldiio==2.15.1
  pip3 install git+https://github.com/huggingface/transformers.git@d5b82bb70c2e8c4b184a6f2a7d1c91d7fd156956
  pip3 install resampy==0.2.2
  pip3 install ConfigArgParse==1.5.1
  pip3 install librosa==0.8.1
  pip3 install scipy==1.7.1
  pip3 install soundfile
  pip3 install amfm_decompy==1.0.11
  # HACK PATCHING pYAAPT.py
  cp .pYAAPT.py ./venv/lib/python3.8/site-packages/amfm_decompy/pYAAPT.py
  pip3 install matplotlib
  pip3 install ffmpeg==1.4
  pip3 install tqdm


  # sidekit additional req
  pip3 install matplotlib==3.4.3
  pip3 install SoundFile==0.10.3.post1
  pip3 install PyYAML==5.4.1
  pip3 install h5py==3.2.1
  pip3 install ipython==7.27.0

  # demo req
  pip3 install ipywebrtc==0.6.0
  pip3 install ipywidgets==7.6.5
  pip3 install notebook==6.4.5
  pip3 install filelock

  cd $home
  touch $mark
fi


mark=.done-kaldi-tools
if [ ! -f $mark ]; then
  echo " == Building Kaldi tools =="
  rm -rf kaldi || true
  git clone https://github.com/kaldi-asr/kaldi.git || true
  cd kaldi
  git checkout 9d235864c
  echo " === Applying personal patch on kaldi ==="
  git apply ../kaldi.patch
  cd tools
  extras/check_dependencies.sh || exit 1
  make -j $nj || exit 1
  cd $home
  touch $mark
fi

mark=.done-kaldi-src
if [ ! -f $mark ]; then
  echo " == Building Kaldi src =="
  cd kaldi/src
  ./configure --shared --use-cuda=yes --mathlib=ATLAS --cudatk-dir=$CUDAROOT || exit 1
  make clean || exit 1
  make depend -j $nj || exit 1
  make -j $nj || exit 1
  cd $home
  touch $mark
fi

export KALDI_ROOT=$home/kaldi
mark=.done-pkwrap
if [ ! -f $mark ]; then
  echo " == Building pkwrap src =="
  # rm -rf pkwrap
  # git clone https://github.com/idiap/pkwrap.git
  cd pkwrap
  # git checkout ccf4094
	python3 setup.py install
	pip3 install -e .
  cd $home
  touch $mark
fi

mark=.done-sidekit
if [ ! -f $mark ]; then
  echo " == Building sidekit =="
  # rm -rf sidekit
  # git clone https://git-lium.univ-lemans.fr/Larcher/sidekit.git
  cd sidekit
  # git checkout 88f4d2b9
  pip3 install -e .
  cd $home
  touch $mark
fi

mark=.done-anonymization_metrics
if [ ! -f $mark ]; then
  git clone https://gitlab.inria.fr/magnet/anonymization_metrics.git
  cd $home
  pip3 install seaborn
  touch $mark
fi


mark=.done-FEERCI
if [ ! -f $mark ]; then
  git clone https://github.com/feerci/feerci
  cd feerci
  pip install Cython
	pip3 install -e .
  cd $home
  touch $mark
fi

mark=.done-fairseq
if [ ! -f $mark ]; then
    rm -rf fairseq || true

    # FairSeq Commit id when making this PR: `commit 313ff0581561c7725ea9430321d6af2901573dfb`
    # git clone --depth 1 https://github.com/pytorch/fairseq.git
    # TODO(jiatong): to fix after the issue #4035 is fixed in fairseq
    git clone https://github.com/pytorch/fairseq.git
    cd fairseq
    git checkout -b sync_commit 313ff0581561c7725ea9430321d6af2901573dfb
    cd ..
    python3 -m pip install --editable ./fairseq
fi

echo " == Everything got installed successfully =="
