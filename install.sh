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
conda_url=https://repo.anaconda.com/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh

# Cluster dependent install
## Colab
if stat -t /usr/local/lib/*/dist-packages/google/colab > /dev/null 2>&1; then
  touch .in_colab_kaggle
  venv_dir=/usr/local
fi
if test -d /kaggle; then
  # Kaggle support is still in WIP
  touch .in_colab_kaggle
  venv_dir=/opt/conda
fi
if test -f .in_colab_kaggle; then
  # Overwrite current python site-package with miniconda one
  # WARNING THIS break everything on anything other than colab!
  venv_dir=/usr/local/

  # use the same python version as collab one (necessary for the overwrite)
  current_python_version=$(python -c 'import sys; print("py" + str(sys.version_info[0]) + str(sys.version_info[1]) )')
  current_python_version_with_dot=$(python -c 'import sys; print(str(sys.version_info[0]) + "." + str(sys.version_info[1]) )')
  file=$(curl -s -S https://repo.anaconda.com/miniconda/ | grep "$current_python_version" | grep "Linux.*x86_64" | head -n 1 | grep -o '".*"' | tr -d '"')
  conda_url=https://repo.anaconda.com/miniconda/$file

  echo " == Google colab / Kaggle detected, running $current_python_version | Warning: Performing $venv_dir OVERWRITE! =="

  echo "Using local \$CUDAROOT: $CUDAROOT"
  cuda_version=$($CUDAROOT/bin/nvcc --version | grep "Cuda compilation tools" | cut -d" " -f5 | sed s/,//)
  cuda_version_witout_dot=$(echo $cuda_version | xargs | sed 's/\.//')
  echo "Cuda version: $cuda_version_witout_dot"

  torch_version=1.10.2
  torchvision_version=0.11.3
  torchaudio_version=0.10.2
  torch_wheels="https://download.pytorch.org/whl/cu$cuda_version_witout_dot/torch_stable.html"

  mark=.done-colab-specific
  if [ ! -f $mark ]; then
    echo " - Downloading a pre-compiled version of kaldi"
    ( # Skip kaldi install
    # And use pre-compiled version (this is not suitable for model training - kaldi GCC/CUDA mismatch with pkwrap)
    curl -L bit.ly/kaldi-colab | tar xz -C / --exclude='usr*'
    ln -s /opt/kaldi/ kaldi
    touch .done-kaldi-tools
    touch .done-kaldi-src
    ) &

    # Cleanup before install
    echo " - Removing some dist-packages/deps before backup"
    \rm -rf /usr/local/cuda-11.0 || true
    \rm -rf /usr/local/cuda-10.1 || true
    \rm -rf /usr/local/cuda-10.0 || true
    for pkg in torch tensorflow plotly cupy ideep4py jaxlib pystan caffe2 music21 xgboost; do
      \rm -rf $venv_dir/lib/python$current_python_version_with_dot/dist-packages/$pkg || true
    done
    \rm -rf /tensorflow-* || true
    \rm -rf /opt/nvidia || true
    # Backup some CUDA before the miniconda overwrite install
    mkdir -p /tmp/backup
    echo " - CUDA /usr/local backup before overwrite"
    cp -r $venv_dir/cuda* /tmp/backup/ || true
    echo " - Python dist-package /usr/local backup before overwrite"
    # Backup dist-packages
    mkdir -p /tmp/backup/lib/python$current_python_version_with_dot/dist-packages
    cp -r $venv_dir/lib/python$current_python_version_with_dot/dist-packages/* \
      /tmp/backup/lib/python$current_python_version_with_dot/dist-packages || true
    wait # wait for kaldi download
    touch $mark
  fi
fi

## Grid5000
if [ "$(id -n -g)" == "g5k-users" ]; then # Grid 5k Cluster (Cuda 11.3 compatible cards (A40))
  echo "Installing on Grid5000, check your GPU (for this node) compatibility with CUDA 11.3!"
  module_load="source /etc/profile.d/lmod.sh"
  eval "$module_load"
  echo "$module_load" >> env.sh
  module_load="module load cuda/11.3.1_gcc-8.3.0"
  eval "$module_load"
  echo "$module_load" >> env.sh
  module_load="module load gcc/8.3.0_gcc-8.3.0"
  eval "$module_load"
  echo "$module_load" >> env.sh
  CUDAROOT=$(which nvcc | head -n1 | xargs | sed 's/\/bin\/nvcc//g')
  yes | sudo-g5k apt install python2.7
  echo "Using local \$CUDAROOT: $CUDAROOT"
  cuda_version=$($CUDAROOT/bin/nvcc --version | grep "Cuda compilation tools" | cut -d" " -f5 | sed s/,//)
  cuda_version_witout_dot=$(echo $cuda_version | xargs | sed 's/\.//')
  echo "Cuda version: $cuda_version_witout_dot"

  torch_version=1.10.2
  torchvision_version=0.11.3
  torchaudio_version=0.10.2
  torch_wheels="https://download.pytorch.org/whl/cu$cuda_version_witout_dot/torch_stable.html"
fi
## Lium
if [ "$(id -g --name)" == "lium" ]; then # LIUM Cluster
  echo "Installing on Lium, check your GPU (for this node) compatibility with CUDA 11.5!"
  CUDAROOT=/opt/cuda/11.5
  echo "Using local \$CUDAROOT: $CUDAROOT"
  cuda_version=$($CUDAROOT/bin/nvcc --version | grep "Cuda compilation tools" | cut -d" " -f5 | sed s/,//)
  cuda_version_witout_dot=$(echo $cuda_version | xargs | sed 's/\.//')
  echo "Cuda version: $cuda_version_witout_dot"

  torch_version=1.11.0
  torchvision_version=0.12.0
  torchaudio_version=0.11.0
  torch_wheels="https://download.pytorch.org/whl/cu$cuda_version_witout_dot/torch_stable.html"
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
  . $venv_dir/bin/activate

  if test -f .in_colab_kaggle; then
    # add back colab deleted /usr/local dependencies
    cp -r /tmp/backup/* $venv_dir
    \rm -rf /tmp/backup/
  fi

  echo "Installing conda dependencies"
  yes | conda install -c conda-forge \
    sox \
    libflac \
    inotify-tools \
    git-lfs \
    ffmpeg \
    wget \
    mkl mkl-include \
    cmake

    # CHECK the cudnn version -> must be compatible with CUDA_HOME version
    # In 2022 cudnn-8.2.1.32 compatible with (cuda 10.2, 11.3... and more)
    # --no-deps to avoid isntalling cudatoolkit (using local cuda at CUDA_HOME)
    yes | conda install -c conda-forge cudnn=8.2.1.32 --no-deps

  touch $mark
fi
source $venv_dir/bin/activate

export PATH=$CUDAROOT/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$venv_dir/lib/:$CUDAROOT/lib64
export CFLAGS="-I$CUDAROOT/include $CFLAGS"
export CUDA_HOME=$CUDAROOT
export CUDA_PATH=$CUDAROOT

export CUDNN_ROOT="$venv_dir"
export CUDNN_INCLUDE_DIR="$venv_dir/include"
export CUDNN_LIBRARY="$venv_dir/lib/libcudnn.so"

export OPENFST_PATH=$(realpath .)/kaldi/tools/openfst
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$OPENFST_PATH/lib

export CPPFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
export CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"

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

  \rm requirements.txt || true
  echo numpy==1.20 >> requirements.txt # force numpy version to 1.20 (required by Numba)

  echo scikit-learn==0.24.2 >> requirements.txt
  echo tensorboard >> requirements.txt
  echo carbontracker==1.1.6 >> requirements.txt
  echo python-dateutil >> requirements.txt

  # pkwrap additional req
  echo pytorch-memlab==0.2.3 >> requirements.txt
  echo kaldiio==2.15.1 >> requirements.txt
  echo git+https://github.com/huggingface/transformers.git@d5b82bb70c2e8c4b184a6f2a7d1c91d7fd156956 >> requirements.txt
  echo resampy==0.2.2 >> requirements.txt
  echo ConfigArgParse==1.5.1 >> requirements.txt
  echo librosa==0.8.1 >> requirements.txt
  echo scipy==1.7.1 >> requirements.txt
  echo amfm_decompy==1.0.11 >> requirements.txt
  echo ffmpeg==1.4 >> requirements.txt
  echo tqdm >> requirements.txt

  # sidekit additional req
  echo matplotlib==3.4.3 >> requirements.txt
  echo SoundFile==0.10.3.post1 >> requirements.txt
  echo PyYAML==5.4.1 >> requirements.txt
  echo h5py==3.2.1 >> requirements.txt
  echo ipython==7.27.0 >> requirements.txt

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
  # git checkout 05f66603a
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
  # ./configure --shared --use-cuda=yes --mathlib=MKL --mkl-root=$MKL_ROOT --cudatk-dir=$CUDAROOT || exit 1
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
  cd pkwrap
  make clean
  python3 setup.py install
  pip3 install -e .
  cd $home
  touch $mark
fi


mark=.done-python-requirements-kaldi-feat
if [ ! -f $mark ]; then
  echo " == Building kaldifeat =="
  \rm -rf kaldifeat || true
  git clone https://github.com/csukuangfj/kaldifeat kaldifeat
  cd kaldifeat
  git checkout cec876b
  export KALDIFEAT_CMAKE_ARGS="-DCUDNN_LIBRARY=$CUDNN_LIBRARY -DCMAKE_BUILD_TYPE=Release -Wno-dev"
  export KALDIFEAT_MAKE_ARGS="-j $nj"
  which python3
  LDFLAGS="-L$venv_dir/lib" python setup.py install || exit 1
  cd $home
  python3 -c "import kaldifeat; print('Kaldifeat version:', kaldifeat.__version__)" || exit 1
  touch $mark
fi

mark=.done-sidekit
if [ ! -f $mark ]; then
  echo " == Building sidekit =="
  if [ ! -d sidekit ]; then
    git clone https://git-lium.univ-lemans.fr/speaker/sidekit sidekit
  fi
  cd sidekit
  # git checkout 70d68c2
  pip3 install -e .
  cd $home
  touch $mark
fi

mark=.done-anonymization_metrics
if [ ! -f $mark ]; then
  echo " == Building anonymization_metrics =="
  rm -rf anonymization_metrics || true
  git clone https://gitlab.inria.fr/magnet/anonymization_metrics.git
  cd anonymization_metrics
  # git checkout 4787d4f
  cd $home
  pip3 install seaborn
  touch $mark
fi


mark=.done-FEERCI
if [ ! -f $mark ]; then
  echo " == Building feerci =="
  rm -rf feerci || true
  git clone https://github.com/feerci/feerci
  cd feerci
  # git checkout 12b5fed
  pip install Cython
  pip3 install -e .
  cd $home
  touch $mark
fi

mark=.done-fairseq
if [ ! -f $mark ]; then
  echo " == Building fairseq =="
    rm -rf fairseq || true
    git clone https://github.com/pytorch/fairseq.git
    cd fairseq
    git checkout -b sync_commit 313ff0581561c7725ea9430321d6af2901573dfb
    cd ..
    python3 -m pip install --editable ./fairseq
    touch $mark
fi


echo "source $venv_dir/bin/activate; export CUDAROOT=$CUDAROOT; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH;" >> env.sh
echo "export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python;" >> env.sh # WORKING around https://github.com/protocolbuffers/protobuf/issues/10051

echo " == Everything got installed successfully =="
