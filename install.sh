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
  touch .in_colab
fi
if test -f .in_colab; then
  # Overwrite current python site-package with miniconda one
  venv_dir=/usr/local/

  # use the same python version as collab one (necessary for the overwrite)
  current_python_version=$(python -c 'import sys; print("py" + str(sys.version_info[0]) + str(sys.version_info[1]) )')
  current_python_version_with_dot=$(python -c 'import sys; print(str(sys.version_info[0]) + "." + str(sys.version_info[1]) )')
  file=$(curl -s -S https://repo.anaconda.com/miniconda/ | grep "$current_python_version" | grep "x86_64" | head -n 1 | grep -o '".*"' | tr -d '"')
  conda_url=https://repo.anaconda.com/miniconda/$file

  echo " == Google colab detected, running $current_python_version =="

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
    # Skip kaldi install
    touch .done-kaldi-tools
    touch .done-kaldi-src
    # And use pre-compiled version (this is not suitable for model training - kaldi GCC/CUDA mismatch with pkwrap)
    curl -L bit.ly/kaldi-colab | tar xz -C /
    ln -s /opt/kaldi/ kaldi

    # Backup some stuff before the miniconda overwrite install
    echo " - CUDA /usr/local backup before overwrite"
    mkdir -p /tmp/backup
    # Remove deps version
    \rm -rf /usr/local/cuda-10.1 || true
    \rm -rf /usr/local/cuda-10.0 || true
    \rm -rf /usr/local/cuda-11.0 || true
    \rm -rf /tensorflow-* || true
    \rm -rf /opt/nvidia || true
    cp -r /usr/local/cuda* /tmp/backup/
    # Backup google.colab library
    clobab_package=$(python -c 'import google; print(str(list(google.__path__)[0]).replace("/usr/local/", ""))' )
    mkdir -p /tmp/backup/$clobab_package
    cp -r /usr/local/$clobab_package/* /tmp/backup/$clobab_package
    touch $mark
  fi
fi

## Grid5000
if [ "$(id -n -g)" == "g5k-users" ]; then # Grid 5k Cluster
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
  CUDAROOT=/opt/cuda/10.2
  echo "Using local \$CUDAROOT: $CUDAROOT"
  cuda_version=$($CUDAROOT/bin/nvcc --version | grep "Cuda compilation tools" | cut -d" " -f5 | sed s/,//)
  cuda_version_witout_dot=$(echo $cuda_version | xargs | sed 's/\.//')
  echo "Cuda version: $cuda_version_witout_dot"

  torch_version=1.8.2
  torchvision_version=0.9.2
  torchaudio_version=0.8.2
  torch_wheels="https://download.pytorch.org/whl/lts/1.8/torch_lts.html"
fi

mark=.done-venv
if [ ! -f $mark ]; then
  echo " == Making python virtual environment =="
  name=$(basename $conda_url)
  if [ ! -f $name ]; then
    wget $conda_url || exit 1
  fi
  [ ! -f $name ] && echo "File $name does not exist" && exit 1
  [ -d $venv_dir ] && rm -r $venv_dir
  sh $name -b -u -p $venv_dir || exit 1
  . $venv_dir/bin/activate

  if test -f .in_colab; then
    # add back colab deleted /usr/local dependencies
    cp -r /tmp/backup/* /usr/local
    \rm -rf /tmp/backup/
    pip install -q --upgrade ipython
    pip install -q --upgrade ipykernel
  fi

  echo "Installing conda dependencies"
  yes | conda install -c conda-forge sox
  yes | conda install -c conda-forge libflac
  yes | conda install -c conda-forge inotify-tools
  touch $mark
fi
source $venv_dir/bin/activate

export PATH=$CUDAROOT/bin:$PATH
export LD_LIBRARY_PATH=$CUDAROOT/lib64:$LD_LIBRARY_PATH:$venv_dir/lib/
export CFLAGS="-I$CUDAROOT/include $CFLAGS"
export CUDA_HOME=$CUDAROOT
export CUDA_PATH=$CUDAROOT

export OPENFST_PATH=$(realpath .)/kaldi/tools/openfst
export LD_LIBRARY_PATH=$OPENFST_PATH/lib:$LD_LIBRARY_PATH

export CPPFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
export CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"

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
  pip3 install python-dateutil

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
  cp .pYAAPT.py $(python3 -c "import amfm_decompy.pYAAPT; print(amfm_decompy.__path__[0])")/pYAAPT.py
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
  make clean
	python3 setup.py install
	pip3 install -e .
  # make test
  cd $home
  touch $mark
fi


mark=.done-python-requirements-kaldi-feat
if [ ! -f $mark ]; then

  if [ "$(id -g --name)" == "lium" ]; then # LIUM Cluster
    yes | conda install -c conda-forge cmake
    yes | conda install -c conda-forge cudnn

    export CUDNN_ROOT="$venv_dir"
    export CUDNN_INCLUDE_DIR="$venv_dir/include"
    export CUDNN_LIBRARY="$venv_dir/lib/libcudnn.so"
    export KALDIFEAT_CMAKE_ARGS="-DCUDNN_LIBRARY=$CUDNN_LIBRARY -DCMAKE_BUILD_TYPE=Release"
    export KALDIFEAT_MAKE_ARGS="-j"
  fi

  if [ "$(id -n -g)" == "g5k-users" ]; then # Grid 5k Cluster
    export LD_LIBRARY_PATH=/grid5000/spack/opt/spack/linux-debian10-x86_64/gcc-8.3.0/gcc-11.1.0-d7x3xputfzupgabmj3hcqis6g4mdpulx/lib64:$LD_LIBRARY_PATH
  fi
  pip3 install kaldifeat==1.12
  cd $home
  python3 -c "import kaldifeat; print('Kaldifeat version:', kaldifeat.__version__)" || exit 1
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
  rm -rf anonymization_metrics || true
  git clone https://gitlab.inria.fr/magnet/anonymization_metrics.git
  cd $home
  pip3 install seaborn
  touch $mark
fi


mark=.done-FEERCI
if [ ! -f $mark ]; then
  rm -rf feerci || true
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
    touch $mark
fi


echo "source $venv_dir/bin/activate; export CUDAROOT=$CUDAROOT; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH;" >> env.sh

echo " == Everything got installed successfully =="
