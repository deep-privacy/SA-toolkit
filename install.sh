#!/bin/bash

set -e

nj=$(nproc)

home=$PWD

conda_url=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
conda_url=https://repo.anaconda.com/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh
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

  echo 'Installing conda dependencies'
  yes | conda install -c conda-forge sox
  yes | conda install -c conda-forge libflac
  touch $mark
fi
echo "if [ \$(which python) != $venv_dir/bin/python ]; then source $venv_dir/bin/activate; fi" > env.sh

source $venv_dir/bin/activate

export CPPFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
export CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"

mark=.done-kaldi-tools
if [ ! -f $mark ]; then
  echo 'Building Kaldi tools'
  # rm -rf kaldi
  git clone https://github.com/kaldi-asr/kaldi.git || true
  # git checkout d619890
  cd kaldi/tools
  extras/check_dependencies.sh || exit 1
  make -j $nj || exit 1
  cd $home
  touch $mark
fi

mark=.done-kaldi-src
if [ ! -f $mark ]; then
  echo 'Building Kaldi src'
  cd kaldi/src
  ./configure --shared --use-cuda=yes --mathlib=ATLAS || exit 1
  make clean || exit 1
  make depend -j $nj || exit 1
  make -j $nj || exit 1
  cd $home
  touch $mark
fi

mark=.done-pytorch
if [ ! -f $mark ]; then
  # pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
  pip3 install torch==1.8.2+cu102 torchvision==0.9.2+cu102 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
  pip install tensorboard
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
  make
  pip install -e .
  pip install umap-learn==0.5.1
  cd $home
  touch $mark
fi

mark=.done-damped
if [ ! -f $mark ]; then
  echo 'Installing damped'
  # rm -rf damped
  # git clone https://github.com/deep-privacy/damped
  cd damped
  pip install -e .
  pip install ConfigArgParse==1.5.1
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
  pip install matplotlib==3.4.3
  pip install SoundFile==0.10.3.post1
  pip install PyYAML==5.4.1
  cd $home
  touch $mark
fi


echo 'Everything got installed succefluly'
