#!/bin/bash

set -e

nj=$(nproc)

home=$PWD
\rm env.sh 2> /dev/null || true
touch env.sh

# CUDA version
CUDAROOT=/usr/local/cuda

# VENV install dir
venv_dir=$PWD/venv

# CONDA
conda_url=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
conda_url=https://repo.anaconda.com/miniconda/Miniconda3-py39_22.11.1-1-Linux-x86_64.sh


torch_version=1.12.1
nightly=

# Cluster dependent installs #

## Colab ##
if stat -t /usr/local/lib/*/dist-packages/google/colab > /dev/null 2>&1; then
  touch .in_colab_kaggle
  # Overwrite current python site-package with miniconda one
  # WARNING THIS break everything on anything other than colab!
  venv_dir=/usr/local/

  # use the same python version as collab one (necessary for the overwrite)
  current_python_version=$(python -c 'import sys; print("py" + str(sys.version_info[0]) + str(sys.version_info[1]) )')
  current_python_version_with_dot=$(python -c 'import sys; print(str(sys.version_info[0]) + "." + str(sys.version_info[1]) )')
  file=$(curl -s -S https://repo.anaconda.com/miniconda/ | grep "$current_python_version" | grep "Linux.*x86_64" | head -n 1 | grep -o '".*"' | tr -d '"')
  conda_url=https://repo.anaconda.com/miniconda/$file

  echo " == Google colab detected, running python $current_python_version_with_dot | WARNING: Performing $venv_dir OVERWRITE! =="
  mark=.done-colab-specific
  if [ ! -f $mark ]; then
    echo " - Downloading a pre-compiled version of kaldi"
    ( # Skip kaldi install
    # And use pre-compiled version (this is not suitable for model training - kaldi GCC/CUDA mismatch with pkwrap)
    curl -L bit.ly/kaldi-colab | tar xz -C / --exclude='usr*'
    ln -s /opt/kaldi/ kaldi
    touch .done-kaldi-src .done-kaldi-tools
    ) &

    # Cleanup before install
    echo " - Removing some dist-packages/deps before backup"
    \rm -rf /opt/nvidia /tensorflow-* /usr/local/cuda-10.0 /usr/local/cuda-10.1 /usr/local/cuda-11.0 || true
    for pkg in torch tensorflow plotly cupy ideep4py jaxlib pystan caffe2 music21 xgboost; do
      \rm -rf $venv_dir/lib/python$current_python_version_with_dot/dist-packages/$pkg || true
    done
    # Backup some CUDA before the miniconda overwrite install
    echo " - CUDA /usr/local backup before overwrite"
    mkdir -p /tmp/backup; cp -r $venv_dir/cuda* /tmp/backup/ || true
    echo " - Python dist-package /usr/local backup before overwrite"
    # Backup dist-packages
    mkdir -p /tmp/backup/lib/python$current_python_version_with_dot/dist-packages
    cp -r $venv_dir/lib/python$current_python_version_with_dot/dist-packages/* \
      /tmp/backup/lib/python$current_python_version_with_dot/dist-packages || true
    wait # wait for kaldi download
    touch $mark
  fi
fi

mark=.done-venv
if [ ! -f $mark ]; then
  echo " == Making python virtual environment =="
  name=$(basename $conda_url)
  if [ ! -f $name ]; then
    wget $conda_url || exit 1
  fi
  [ ! -f $name ] && echo "File $name does not exist" && exit 1
  [ -d $venv_dir ] && yes | rm -rf $venv_dir
  bash $name -b -u -p $venv_dir || exit 1
  source $venv_dir/bin/activate ''

  if test -f .in_colab_kaggle; then
    # add back colab deleted /usr/local dependencies
    cp -r /tmp/backup/* $venv_dir; \rm -rf /tmp/backup/
    \rm -f $name || true
  fi

  echo "Installing conda dependencies"
  yes | conda install -c conda-forge \
    sox libflac inotify-tools git-lfs ffmpeg wget mkl mkl-include cmake ncurses

  touch $mark
fi
source $venv_dir/bin/activate ''




# Cluster dependent installs #
## Lium ##
if [ "$(id -g --name)" == "lium" ]; then # LIUM Cluster
  echo "Installing on Lium"
  mark=.done-cuda
  if [ ! -f $mark ]; then
    yes | conda install -c "nvidia/label/cuda-11.7.0" cuda-toolkit
    touch $mark
  fi

  CUDAROOT=$venv_dir
  torch_version=2.0.0
  nightly='nightly/'
fi

cuda_version=$($CUDAROOT/bin/nvcc --version | grep "Cuda compilation tools" | cut -d" " -f5 | sed s/,//)

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -c|--cuda) cuda_version="$2"; shift; echo "Arg Cuda $cuda_version" ;;
        -t|--torch) torch_version=$2; shift; echo "Arg Torch $torch_version" ;;
        -n|--nightly) nightly='nightly/'; echo "Arg nightly whl" ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

cuda_version_witout_dot=$(echo $cuda_version | xargs | sed 's/\.//')
echo "Local \$CUDAROOT: $CUDAROOT with cuda version: $cuda_version"

export PATH=$CUDAROOT/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$venv_dir/lib/:$CUDAROOT/lib64
export CFLAGS="-I$CUDAROOT/include $CFLAGS"
export CUDA_HOME=$CUDAROOT
export CUDA_PATH=$CUDAROOT

export OPENFST_PATH=$(realpath .)/kaldi/tools/openfst
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$OPENFST_PATH/lib

export CPPFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
export CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"

mark=.done-pytorch
if [ ! -f $mark ]; then
  echo " == Installing pytorch $torch_version for cuda $cuda_version =="
  version="==$torch_version+cu$cuda_version_witout_dot"
  pre= ; if [[ "$nightly" == "nightly/" ]]; then pre="--pre"; version=""; fi
  echo -e "\npip3 install $pre torch$version torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/${nightly}cu$cuda_version_witout_dot\n"
  pip3 install $pre torch$version torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/${nightly}cu$cuda_version_witout_dot \
    || { echo "Failed to find pytorch $torch_version for cuda '$cuda_version', use please specify another version with: './install.sh --cuda 11.3 --torch 1.12.1' --nightly" ; exit 1; }
  cd $home
  touch $mark
fi


mark=.done-python-requirements
if [ ! -f $mark ]; then
  echo " == Installing python libraries =="

  pip install Cython

  \rm requirements.txt || true
  echo numpy==1.20.3 >> requirements.txt # force numpy version to 1.20.3 (required by Numba and pandas)

  echo scikit-learn==0.24.2 >> requirements.txt
  echo tensorboard >> requirements.txt
  echo carbontracker==1.1.6 >> requirements.txt
  echo python-dateutil >> requirements.txt

  # asr additional req
  echo pytorch-memlab==0.2.3 >> requirements.txt
  echo kaldiio==2.15.1 >> requirements.txt
  echo resampy==0.2.2 >> requirements.txt
  echo ConfigArgParse==1.5.1 >> requirements.txt
  echo librosa==0.8.1 >> requirements.txt
  echo scipy==1.7.1 >> requirements.txt
  echo amfm_decompy==1.0.11 >> requirements.txt
  echo ffmpeg==1.4 >> requirements.txt
  echo tqdm >> requirements.txt
  # echo 'git+https://github.com/pytorch/fairseq.git@313ff0581561c7725ea9430321d6af2901573dfb' >> requirements.txt

  # sidekit additional req
  echo matplotlib==3.4.3 >> requirements.txt
  echo SoundFile==0.10.3.post1 >> requirements.txt
  echo PyYAML==5.4.1 >> requirements.txt
  echo h5py==3.2.1 >> requirements.txt
  echo ipython==7.27.0 >> requirements.txt
  echo 'git+https://github.com/feerci/feerci' >> requirements.txt
  echo 'pandas>=1.0.5' >> requirements.txt

  # demo req
  echo ipywebrtc==0.6.0 >> requirements.txt
  echo ipywidgets==7.6.5 >> requirements.txt
  echo notebook==6.4.5 >> requirements.txt
  echo filelock >> requirements.txt


  pip3 install -r requirements.txt

  # HACK PATCHING pYAAPT.py
  cp .pYAAPT.py $(python3 -c "import amfm_decompy.pYAAPT; print(amfm_decompy.__path__[0])")/pYAAPT.py

  cd $home
  touch $mark
fi


mark=.done-kaldi-tools
if [ ! -f $mark ]; then
  echo " == Building Kaldi tools =="
  rm -rf kaldi || true
  git clone https://github.com/kaldi-asr/kaldi.git || true
  cd kaldi
  git checkout e4eb4f6
  echo " === Applying personal patch on kaldi ==="
  git apply ../.kaldi.patch
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
  if test -f .in_colab_kaggle; then
    export PKWRAP_CPP_EXT=no
  fi
  cd satools
  make clean
  python3 setup.py install
  pip3 install -e .
  cd $home
  touch $mark
fi

# export GIT_SSL_NO_VERIFY=1
# mark=.done-sidekit
# if [ ! -f $mark ]; then
  # echo " == Building sidekit =="
  # if [ ! -d sidekit ]; then
    # git clone https://git-lium.univ-lemans.fr/speaker/sidekit sidekit
  # fi
  # cd sidekit
  # # git checkout 70d68c2
  # pip3 install -e .
  # cd $home
  # touch $mark
# fi

echo "source $venv_dir/bin/activate ''; export CUDAROOT=$CUDAROOT; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH;" >> env.sh
echo "export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python;" >> env.sh # WORKING around https://github.com/protocolbuffers/protobuf/issues/10051

echo " == Everything got installed successfully =="
