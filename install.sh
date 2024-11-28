#!/bin/bash

# Fresh install with "rm .micromamba/micromamba .*done-*"

set -e

nj=$(nproc)


# VENV install dir / name / mamba_root_prefix

venv_name=env.sh
venv_dir_name=venv

# Local install of micromamba (where the libs/bin will be cached)
# This can be share with other mamba install,
# best to be modified when there is multiple users and installs!
mamba_root_prefix="$HOME/micromamba"


### VERSIONS ####

MAMBA_VERSION=2.0.2-0
PYTHON_VERSION=3.11
GCC_VERSION=12.3.0

CUDA_VERSION=12.1
TORCH_VERSION=2.1.2

MAMBA_PACKAGES_TO_INSTALL="sshpass OpenSSH sox libflac tar libacl inotify-tools ocl-icd-system git-lfs ffmpeg wget curl make cmake ncurses ninja python=$PYTHON_VERSION htop nvtop automake libtool boost gxx=$GCC_VERSION gcc=$GCC_VERSION python-sounddevice pkg-config gzip zip unzip patch zlib libzlib gfortran"

INSTALL_KALDI=true
KALDI_SHORT_ID="4a8b7f673" # commit id of kaldi (or empty for master)


home=$PWD
\rm $venv_name 2> /dev/null || true
touch $venv_name
venv_dir=$PWD/$venv_dir_name

export MAMBA_ROOT_PREFIX="$mamba_root_prefix" 
mamba_bin="$MAMBA_ROOT_PREFIX/micromamba"


explain() {   printf "\033[0;34m${1}\033[0m\n"; }
info() {  printf "\033[0;33m${1}\033[0m\n"; }

### VENV ###

mark=.${venv_dir_name}_done-venv
if [ ! -f $mark ]; then
  explain " == Making virtual environment =="
  if [ ! -f "$mamba_bin" ]; then
    info "Downloading micromamba"
    mkdir -p "$MAMBA_ROOT_PREFIX"
    curl -sS -L "https://github.com/mamba-org/micromamba-releases/releases/download/$MAMBA_VERSION/micromamba-linux-64" > "$mamba_bin"
    chmod +x "$mamba_bin"
  fi
  [ -d $venv_dir ] && yes | rm -rf $venv_dir

  info "Micromamba version: $($mamba_bin --version)"

  "$mamba_bin" config set always_softlink false
  "$mamba_bin" config set allow_softlinks false
  "$mamba_bin" create -y --prefix "$venv_dir"

  info "Installing conda dependencies"
  "$mamba_bin" install -y --prefix "$venv_dir" -c conda-forge $MAMBA_PACKAGES_TO_INSTALL || exit 1
  info "Python version: $($venv_dir/bin/python --version)" || exit 1

  mkdir -p $MAMBA_ROOT_PREFIX/shared-site-packages_torch${TORCH_VERSION}_cu${CUDA_VERSION}
  touch $mark
fi
# "$mamba_bin" install -y --prefix "$venv_dir" -c conda-forge $MAMBA_PACKAGES_TO_INSTALL || exit 1

if [ -e "$venv_dir" ]; then export PATH="$venv_dir/bin:$PATH"; fi

# Hook Micromamba into the script's subshell (this only lasts for as long as the # script is running)
echo "eval \"\$($mamba_bin shell hook --shell=bash)\"" >> $venv_name
echo "micromamba activate $venv_dir" >> $venv_name
echo "export LD_LIBRARY_PATH=$venv_dir/lib/stubs/:$venv_dir/lib/:$LD_LIBRARY_PATH" >> $venv_name
echo "alias conda=micromamba" >> $venv_name
echo "export MAMBA_ROOT_PREFIX=$MAMBA_ROOT_PREFIX" >> $venv_name
echo "export PIP_CACHE_DIR=$MAMBA_ROOT_PREFIX/pip_cache" >> $venv_name
echo "export MAMBA_CUSTOM_PY_DEPS=$MAMBA_ROOT_PREFIX/shared-site-packages_torch${TORCH_VERSION}_cu${CUDA_VERSION}" >> env.sh
echo "nvtop() { LD_LIBRARY_PATH=$venv_dir/lib/stubs/libnvidia-ml.so $venv_dir/bin/nvtop; }" >> $venv_name
echo "export PYTHONPATH=$PYTHONPATH:$MAMBA_ROOT_PREFIX/shared-site-packages_torch${TORCH_VERSION}_cu${CUDA_VERSION}/lib/python$PYTHON_VERSION/site-packages" >> env.sh
echo "export PIP_REQUIRE_VIRTUALENV=false" >> $venv_name
echo "alias mminstall=\"'$mamba_bin' install -y --prefix '$venv_dir' -c conda-forge\"" >> $venv_name
source ./$venv_name


## CUDA toolkit ###

mark=.${venv_dir_name}_done-cuda
if [ ! -f $mark ]; then
  explain " == Installing cuda =="
  micromamba install -y --prefix "$venv_dir" -c "nvidia/label/cuda-${CUDA_VERSION}.0" cuda-toolkit || exit 1
  info "nvcc version:\n$($venv_dir/bin/nvcc --version)" || exit 1
  touch $mark
fi

CUDAROOT=$venv_dir
echo "export CUDAROOT=$CUDAROOT" >> $venv_name
source ./$venv_name


### Torch ###

cuda_version_without_dot=$(echo $CUDA_VERSION | xargs | sed 's/\.//')
mark=.${venv_dir_name}_done-pytorch
if [ ! -f $mark ]; then
  explain " == Installing pytorch $TORCH_VERSION for cuda $CUDA_VERSION =="
  version="==$TORCH_VERSION+cu$cuda_version_without_dot"
  info "Torch install command:\npip3 install torch$version torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/${nightly}cu$cuda_version_without_dot"
  python3 -m pip freeze | grep "torch$version" -q || pip3 install --prefix $MAMBA_CUSTOM_PY_DEPS torch$version torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/${nightly}cu$cuda_version_without_dot \
    || { info "Failed to find pytorch $TORCH_VERSION for cuda '$CUDA_VERSION', use specify other torch/cuda versions (with the variables in install.sh script)"  ; exit 1; }
  info "Torch version: $(python3 -c 'import torch; print(torch.__version__)')" || exit 1
  touch $mark
fi


# Python requirements

mark=.${venv_dir_name}_done-python-requirements
if [ ! -f $mark ]; then
  explain " == Installing python libraries =="

  pip3 install Cython

  \rm requirements.txt || true
  echo 'scikit-learn>=0.24.2' >> requirements.txt
  echo 'tensorboard' >> requirements.txt
  echo 'carbontracker==2.0.0' >> requirements.txt
  echo 'matplotlib' >> requirements.txt
  echo 'python-dateutil' >> requirements.txt
  echo 'graftr' >> requirements.txt # an interactive shell to view and edit PyTorch checkpoints

  # asr additional req
  echo 'kaldiio>=2.15.1' >> requirements.txt
  echo 'resampy>=0.2.2' >> requirements.txt
  echo 'librosa' >> requirements.txt
  echo 'scipy>=1.8' >> requirements.txt
  echo 'ffmpeg>=1.4' >> requirements.txt
  echo 'tqdm' >> requirements.txt
  echo 'safe-gpu' >> requirements.txt

  # sidekit additional req
  echo 'git+https://github.com/deep-privacy/feerci@dev' >> requirements.txt
  echo 'pandas>=1.0.5' >> requirements.txt

  pip3 install -r requirements.txt

  cd $home
  touch $mark
fi


# EXPORTS for C++ compilation

export PATH=$CUDAROOT/bin:$PATH

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$venv_dir/lib/:$CUDAROOT/lib64
export CFLAGS="-I$CUDAROOT/include $CFLAGS"
export CUDA_HOME=$CUDAROOT
export CUDA_PATH=$CUDAROOT

export OPENFST_PATH=$(realpath .)/kaldi/tools/openfst
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$OPENFST_PATH/lib

export KALDI_ROOT=$(realpath .)/kaldi
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$KALDI_ROOT/src/lib

export CPPFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0 -I$venv_dir/include -L$LD_LIBRARY_PATH -std=c++17"
export CXXFLAGS="$CPPFLAGS"


### Kaldi tools ###

mark=.${venv_dir_name}_done-kaldi-tools
if [ ! -f $mark ]; then
  explain " == Building Kaldi tools =="
  rm -rf kaldi || true
  git clone https://github.com/kaldi-asr/kaldi.git --branch master --single-branch || true
  cd kaldi
  [[ -v KALDI_SHORT_ID ]] && git checkout ${KALDI_SHORT_ID}
  info "Applying personal patch on kaldi"
  git apply ../.kaldi.patch


  if [ ! $INSTALL_KALDI = true ]; then
    explain " == No futher installation of kaldi, skipping C++ compilation ==" # still need utils/local dirs from kaldi
    cd $home
    touch .${venv_dir_name}_done-kaldi-tools .${venv_dir_name}_done-kaldi-src
  else
    mkdir -p kaldi/tools/python
    touch kaldi/tools/python/.use_default_python
    cd tools
    extras/check_dependencies.sh || exit 1
    make -j $nj || exit 1

    extras/install_openblas.sh || exit 1

    # # Installing srilm to modify language models.
    # # Modifiying installation script. Original one can be find under : kaldi/tools/extras/install_srilm.sh
    # sed -i -e "s|wget.*srilm_url.*$|wget -O ./srilm.tar.gz 'https://github.com/BitSpeech/SRILM/archive/refs/tags/1.7.3.tar.gz';then|g" install_srilm.sh
    # sed -i -e "s|tar -xvzf ../srilm.tar.gz|tar -xvzf ../srilm.tar.gz --strip-components=1|g" install_srilm.sh
    # sed -i -e "s|$venv_name|$home/$venv_name|g" install_srilm.sh
    # # Running installation with fake arguments to bypass argument checking
    # ./install_srilm.sh x x x x

    cd $home
    touch $mark
  fi
fi


### Kaldi SRC ###

mark=.${venv_dir_name}_done-kaldi-src
if [ ! -f $mark ]; then
  explain " == Building Kaldi src =="
  cd kaldi/src
  ./configure --shared --use-cuda=yes --cudatk-dir=$CUDAROOT --mathlib=OPENBLAS  || exit 1
  make clean || exit 1
  make depend -j $nj || exit 1
  make -j $nj || exit 1
  cd $home
  find ./kaldi  -type f \( -name "*.o" -o -name "*.la" -o -name "*.a" \) -exec rm {} \;
  touch $mark
fi


### SA-tools pip/c++ install ###

export OPENFSTVER=$(cat kaldi/src/kaldi.mk| grep "^OPENFSTVER\s*=" | tr -d ' ' | tr -d 'OPENFSTVER=')
export CPPFLAGS="$CPPFLAGS -DOPENFST_VER=$OPENFSTVER"
export CXXFLAGS="$CPPFLAGS"

mark=.${venv_dir_name}_done-satools
if [ ! -f $mark ]; then
  explain " == Building satools src =="
  cd satools
  if [ $INSTALL_KALDI = true ]; then
    explain " == Installing the kaldi binding for LF-MMI ASR (ASR-BN) training =="
    export PKWRAP_CPP_EXT=yes
  fi
  make cleanly
  cd $home
  touch $mark
fi

echo "export CUDAROOT=$CUDAROOT; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH;" >> $venv_name

explain " == Everything got installed successfully =="
