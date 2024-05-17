#!/bin/bash

# Fresh install with "rm .micromamba/micromamba .done-*"

set -e

nj=$(nproc)

home=$PWD
\rm env.sh 2> /dev/null || true
touch env.sh

# VENV install dir
venv_dir=$PWD/venv
export MAMBA_ROOT_PREFIX="$home/.micromamba"  # Local install of micromamba (where the libs/bin will be cached)
mamba_bin="$MAMBA_ROOT_PREFIX/micromamba"

### VERSION

MAMBA_VERSION=1.5.1-0

CUDA_VERSION=11.7
TORCH_VERSION=2.0.1

MAMBA_PACKAGES_TO_INSTALL="sshpass OpenSSH sox libflac tar libacl inotify-tools ocl-icd-system git-lfs ffmpeg wget curl make cmake ncurses ninja python=3.11 nvtop automake libtool boost gxx=12.3.0 gcc=12.3.0 python-sounddevice pkg-config zip openblas zlib"

INSTALL_KALDI=false



mark=.done-venv
if [ ! -f $mark ]; then
  echo " == Making virtual environment =="
  if [ ! -f "$mamba_bin" ]; then
    echo "Downloading micromamba"
    mkdir -p "$MAMBA_ROOT_PREFIX"
    curl -sS -L "https://github.com/mamba-org/micromamba-releases/releases/download/$MAMBA_VERSION/micromamba-linux-64" > "$mamba_bin"
    chmod +x "$mamba_bin"
  fi
  [ -d $venv_dir ] && yes | rm -rf $venv_dir

  echo "Micromamba version:"
  "$mamba_bin" --version

  "$mamba_bin" create -y --prefix "$venv_dir"

  echo "Installing conda dependencies"
  "$mamba_bin" install -y --prefix "$venv_dir" -c conda-forge $MAMBA_PACKAGES_TO_INSTALL || exit 1
  "$venv_dir/bin/python" --version || exit 1

  touch $mark
fi

if [ -e "$venv_dir" ]; then export PATH="$venv_dir/bin:$PATH"; fi

# Hook Micromamba into the script's subshell (this only lasts for as long as the # script is running)
echo "eval \"\$($mamba_bin shell hook --shell=bash)\"" >> env.sh
echo "micromamba activate $venv_dir" >> env.sh
echo "export LD_LIBRARY_PATH=$venv_dir/lib/:$LD_LIBRARY_PATH" >> env.sh
echo "alias conda=micromamba" >> env.sh
echo "export PIP_REQUIRE_VIRTUALENV=false" >> env.sh
source ./env.sh

mark=.done-cuda
if [ ! -f $mark ]; then
  echo " == Installing cuda =="
  micromamba install -y --prefix "$venv_dir" -c "nvidia/label/cuda-${CUDA_VERSION}.0" cuda-toolkit || exit 1
  "$venv_dir/bin/nvcc" --version || exit 1
  touch $mark
fi


CUDAROOT=$venv_dir
echo "export CUDAROOT=$CUDAROOT" >> env.sh
source ./env.sh


cuda_version_without_dot=$(echo $CUDA_VERSION | xargs | sed 's/\.//')
mark=.done-pytorch
if [ ! -f $mark ]; then
  echo " == Installing pytorch $TORCH_VERSION for cuda $CUDA_VERSION =="
  version="==$TORCH_VERSION+cu$cuda_version_without_dot"
  echo -e "\npip3 install torch$version torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/${nightly}cu$cuda_version_without_dot\n"
  pip3 install torch$version torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/${nightly}cu$cuda_version_without_dot \
    || { echo "Failed to find pytorch $TORCH_VERSION for cuda '$CUDA_VERSION', use specify other torch/cuda version (with variables in install.sh script)"  ; exit 1; }
  python3 -c "import torch; print('Torch version:', torch.__version__)" || exit 1
  touch $mark
fi

export PATH=$CUDAROOT/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$venv_dir/lib/:$CUDAROOT/lib64
export CFLAGS="-I$CUDAROOT/include $CFLAGS"
export CUDA_HOME=$CUDAROOT
export CUDA_PATH=$CUDAROOT

export OPENFST_PATH=$(realpath .)/kaldi/tools/openfst
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$OPENFST_PATH/lib

export CPPFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
export CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"

mark=.done-python-requirements
if [ ! -f $mark ]; then
  echo " == Installing python libraries =="

  pip3 install Cython

  \rm requirements.txt || true
  echo 'scikit-learn>=0.24.2' >> requirements.txt
  echo 'tensorboard' >> requirements.txt
  echo 'carbontracker' >> requirements.txt
  echo 'matplotlib' >> requirements.txt
  echo 'python-dateutil' >> requirements.txt
  echo 'graftr' >> requirements.txt # an interactive shell to view and edit PyTorch checkpoints
  echo 'h5py' >> requirements.txt

  # asr additional req
  echo 'kaldiio>=2.15.1' >> requirements.txt
  echo 'resampy>=0.2.2' >> requirements.txt
  echo 'ConfigArgParse==1.5.1' >> requirements.txt
  echo 'librosa' >> requirements.txt
  echo 'scipy>=1.8' >> requirements.txt
  echo 'ffmpeg>=1.4' >> requirements.txt
  echo 'tqdm' >> requirements.txt

  # sidekit additional req
  echo 'git+https://github.com/deep-privacy/feerci@dev' >> requirements.txt
  echo 'pandas>=1.0.5' >> requirements.txt

  # demo req
  echo 'ipywebrtc>=0.6.0' >> requirements.txt
  echo 'ipywidgets>=7.6.5' >> requirements.txt
  echo 'notebook>=6.4.5' >> requirements.txt

  pip3 install -r requirements.txt

  cd $home
  touch $mark
fi

mark=.done-kaldi-tools
if [ ! -f $mark ]; then
  echo " == Building Kaldi tools =="
  rm -rf kaldi || true
  git clone https://github.com/kaldi-asr/kaldi.git || true
  cd kaldi
  # git checkout e4eb4f6
  echo " === Applying personal patch on kaldi ==="
  git apply ../.kaldi.patch


  if [ ! $INSTALL_KALDI = true ]; then
    echo " == No futher installation of kaldi =="
    touch .done-kaldi-tools
    touch .done-kaldi-src
  else
    mkdir -p kaldi/tools/python
    touch kaldi/tools/python/.use_default_python
    cd tools
    extras/check_dependencies.sh || exit 1
    make -j $nj || exit 1

    # Installing srilm to modify language models.
    # Modifiying installation script. Original one can be find under : kaldi/tools/extras/install_srilm.sh
    sed -i -e "s|wget.*srilm_url.*$|wget -O ./srilm.tar.gz 'https://github.com/BitSpeech/SRILM/archive/refs/tags/1.7.3.tar.gz';then|g" install_srilm.sh
    sed -i -e "s|tar -xvzf ../srilm.tar.gz|tar -xvzf ../srilm.tar.gz --strip-components=1|g" install_srilm.sh
    sed -i -e "s|env.sh|$home/env.sh|g" install_srilm.sh
    # Running installation with fake arguments to bypass argument checking
    ./install_srilm.sh x x x
  fi
  cd $home
  touch $mark
fi

mark=.done-kaldi-src
if [ ! -f $mark ]; then
  echo " == Building Kaldi src =="
  cd kaldi/src
  ./configure --shared --use-cuda=yes --mathlib=ATLAS --cudatk-dir=$CUDAROOT || exit 1
  # ./configure --shared --use-cuda=yes --mathlib=MKL --mkl-root=$MKL_ROOT --cudatk-dir=$CUDAROOT || exit 1
  make clean || exit 1
  make depend -j $nj || exit 1
  make -j $nj || exit 1
  cd $home
  touch $mark
fi

export KALDI_ROOT=$home/kaldi
mark=.done-satools
if [ ! -f $mark ]; then
  echo " == Building satools src =="
  cd satools
  if [ ! $INSTALL_KALDI = true ]; then
    echo " == Not installing the kaldi binding for lf-mmi ASR training =="
    export PKWRAP_CPP_EXT=no
  fi
  make cleanly
  cd $home
  touch $mark
fi

echo "export CUDAROOT=$CUDAROOT; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH;" >> env.sh
echo "export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python;" >> env.sh # WORKING around https://github.com/protocolbuffers/protobuf/issues/10051

echo " == Everything got installed successfully =="
