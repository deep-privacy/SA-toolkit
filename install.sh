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
  # make it specific to your cuda version
  # conda search cudnn -c conda-forge
  # conda search cudnn
  cudnn_version="cudnn=7.6.5=cuda10.2_0"
  echo " == CHECK THIS: Installing HARDCODED $cudnn_version!!!!!!!!  =="
  echo " == CHECK THIS: Installing HARDCODED $cudnn_version!!!!!!!!  =="
  sleep 2
  yes | conda install cudnn=7.6.5=cuda10.2_0 || exit 1

  export CUDNN_ROOT="$venv_dir"
  export CUDNN_INCLUDE_DIR="$venv_dir/include"
  export CUDNN_LIBRARY="$venv_dir/lib/libcudnn.so"
  export KALDIFEAT_CMAKE_ARGS="-DCUDNN_LIBRARY=$CUDNN_LIBRARY -DCMAKE_BUILD_TYPE=Release"
  export KALDIFEAT_MAKE_ARGS="-j"

  git clone https://github.com/csukuangfj/kaldifeat
  cd kaldifeat
  git checkout 5e1a9b8
  echo " === Applying personal patch on kaldi ==="
  git apply ../kaldifeat_install.patch
  rm build_release -rf || true
  mkdir build_release
  cd build_release
  cmake .. $KALDIFEAT_CMAKE_ARGS
  make VERBOSE=1 $KALDIFEAT_MAKE_ARGS
  cd ..
  python3 setup.py install
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

  # pkwrap additional req
  pip3 install pytorch-memlab==0.2.3
  pip3 install kaldiio==2.15.1
  pip3 install git+https://github.com/huggingface/transformers.git@d5b82bb70c2e8c4b184a6f2a7d1c91d7fd156956

  # damped additional req
  pip3 install ConfigArgParse==1.5.1

  # sidekit additional req
  pip3 install matplotlib==3.4.3
  pip3 install SoundFile==0.10.3.post1
  pip3 install PyYAML==5.4.1
  pip3 install h5py==3.2.1
  pip3 install ipython==7.27.0

  # speech-resynthesis additional req
  pip3 install librosa==0.8.1
  pip3 install scipy==1.7.1
  pip3 install soundfile
  pip3 install amfm_decompy==1.0.11
  pip3 install matplotlib
  pip3 install ffmpeg==1.4
  pip3 install tqdm

  # demo req
  pip3 install ipywebrtc==0.6.0
  pip3 install ipywidgets==7.6.5
  pip3 install notebook==6.4.5

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

# might use pychain at some point for faster e2e-lfmmi evaluation (need to
# manualy add xent and l2 regularisation)
# have look here:https://github.com/m-wiesner/nnet_pytorch/blob/086bc45cf2f1a12197f29033a1e129f6c8b55b03/nnet_pytorch/objectives/LFMMI.py
# or here: https://github.com/freewym/espresso/blob/65e52fbf4b2aa28809bde7fa6a32bd12ad6d90dc/espresso/criterions/lf_mmi_loss.py
# How to use some part of pychain: https://github.com/lucasondel/MarkovModels.jl/blob/78a70b902c32f814b5c71b52bba72764563fae75/misc/benchmark_pychain.py
# mark=.done-pychain
# if [ ! -f $mark ]; then
  # git clone https://github.com/YiwenShaoStephen/pychain
  # cd pychain
  # git checkout 7bdfdad
  # cd openfst_binding
  # # OPENFST_PATH and LD_LIBRARY_PATH must point to above kaldi/tools
  # python3 setup.py install
  # cd ..
  # cd pytorch_binding
  # python3 setup.py install
  # cd $home
  # echo "export PYTHONPATH=$PYTHONPATH:$(realpath .)/pychain" >> env.sh
  # touch $mark
# fi

mark=.done-damped
if [ ! -f $mark ]; then
  echo " == Building damped =="
  # rm -rf damped
  # git clone https://github.com/deep-privacy/damped
  cd damped
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

mark=.done-speech-resynthesis
if [ ! -f $mark ]; then
  git clone https://github.com/facebookresearch/speech-resynthesis.git
  cd speech-resynthesis
  git checkout a5c97de
  cd -
  touch $mark
fi

echo " == Everything got installed successfully =="
