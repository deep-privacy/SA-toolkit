#!/bin/bash

set -e

nj=$(nproc)

home=$PWD

# CUDA version
CUDAROOT=/usr/local/cuda
if [ "$(id -g --name)" == "lium" ]; then
  CUDAROOT=/opt/cuda/10.1 # LIUM Cluster
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


# Match G5K cuda version
torch_version=1.7.1
torchvision_version=0.8.2
torchaudio_version=0.7.2
torch_wheels="https://download.pytorch.org/whl/torch_stable.html"


venv_dir=$PWD/venv

mark=.done-venv
if [ ! -f $mark ]; then
  echo 'Making python virtual environment'
  name=$(basename $conda_url)
  if [ ! -f $name ]; then
    wget $conda_url || exit 1
  fi
  [ ! -f $name ] && echo "File $name does not exist" && exit 1
  [ -d $venv_dir ] && rm -r $venv_dir
  sh $name -b -p $venv_dir || exit 1
  . $venv_dir/bin/activate

  pip3 install scikit-learn==0.24.2

  echo 'Installing conda dependencies'
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

echo "if [ \$(which python) != $venv_dir/bin/python ]; then source $venv_dir/bin/activate; fi; export CUDAROOT=$CUDAROOT; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH;" > env.sh

export CPPFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
export CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"

mark=.done-kaldi-tools
if [ ! -f $mark ]; then
  echo 'Building Kaldi tools'
  rm -rf kaldi || true
  git clone https://github.com/kaldi-asr/kaldi.git || true
  cd kaldi
  git checkout 9d235864c
  cd tools
  extras/check_dependencies.sh || exit 1
  make -j $nj || exit 1
  cd $home
  touch $mark
fi

mark=.done-kaldi-src
if [ ! -f $mark ]; then
  echo 'Building Kaldi src'
  cd kaldi/src
  ./configure --shared --use-cuda=yes --mathlib=ATLAS --cudatk-dir=$CUDAROOT || exit 1
  make clean || exit 1
  make depend -j $nj || exit 1
  make -j $nj || exit 1
  cd $home
  touch $mark
fi

mark=.done-pytorch
if [ ! -f $mark ]; then
  # pip3 install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
  pip3 install torch==$torch_version+cu$cuda_version_witout_dot torchvision==$torchvision_version+cu$cuda_version_witout_dot torchaudio==$torchaudio_version -f $torch_wheels
  pip3 install tensorboard
  cd $home
  touch $mark
fi

export KALDI_ROOT=$home/kaldi
mark=.done-pkwrap
if [ ! -f $mark ]; then
  echo 'Building pkwrap src'
  # rm -rf pkwrap
  # git clone https://github.com/idiap/pkwrap.git
  cd pkwrap
  # git checkout ccf4094
	python3 setup.py install
	pip3 install -e .
  pip3 install kaldiio==2.15.1
  cd $home
  touch $mark
fi

mark=.done-damped
if [ ! -f $mark ]; then
  echo 'Installing damped'
  # rm -rf damped
  # git clone https://github.com/deep-privacy/damped
  cd damped
  pip3 install -e .
  pip3 install ConfigArgParse==1.5.1
  cd $home
  touch $mark
fi

mark=.done-sidekit
if [ ! -f $mark ]; then
  echo 'Installing sidekit'
  # rm -rf sidekit
  # git clone https://git-lium.univ-lemans.fr/Larcher/sidekit.git
  cd sidekit
  # git checkout 88f4d2b9
  pip3 install matplotlib==3.4.3
  pip3 install SoundFile==0.10.3.post1
  pip3 install PyYAML==5.4.1
  cd $home
  touch $mark
fi


echo 'Everything got installed succefluly'
