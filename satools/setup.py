import os
import sys

from setuptools import setup, find_packages

PKWRAP_CPP_EXT = os.getenv("PKWRAP_CPP_EXT", "no")

KALDI_ROOT = os.getenv("KALDI_ROOT")
if not KALDI_ROOT and PKWRAP_CPP_EXT != "no":
    sys.stderr.write("WARNING: KALDI_ROOT variable is not defined or empty")
    sys.stderr.write("Installing satools without KALDI bindings (ASR training)")
    PKWRAP_CPP_EXT = "no"
    # quit(1)

if PKWRAP_CPP_EXT == "no":
    KALDI_LIB_DIR = "/tmp/fake"
else:
    KALDI_LIB_DIR = os.path.join(KALDI_ROOT, "src", "lib")

PACKAGE_NAME = "satools"
EXTENSION_NAME = "_satools"
SRC_FILES = [
    "csrc/pkwrap-main.cc",
    "csrc/decoder.cc",
    "csrc/matrix.cc",
    "csrc/chain.cc",
    "csrc/nnet3.cc",
    "csrc/fst.cc",
    "csrc/hmm.cc",
]
EXTRA_COMPILE_ARGS = {
    "cxx": [
        "-I{}/src".format(KALDI_ROOT),
        "-I{}/tools/openfst/include".format(KALDI_ROOT),
        "-m64",
        "-msse",
        "-msse2",
        "-DHAVE_CUDA=1",
        # this helped clear openfst related compilation errors
        "-Wno-sign-compare",
        # additional flags used by Kaldi, but not sure we need this
        "-Wno-deprecated-declarations",
        "-Winit-self",
        "-DKALDI_DOUBLEPRECISION=0",
        "-DHAVE_EXECINFO_H=1",
        "-w",  # potentially dangerous, but less annoying
    ]
}
LIBRARIES = [
    "kaldi-base",
    "kaldi-matrix",
    "kaldi-util",
    "kaldi-cudamatrix",
    "kaldi-decoder",
    "kaldi-lat",
    "kaldi-gmm",
    "kaldi-hmm",
    "kaldi-tree",
    "kaldi-transform",
    "kaldi-chain",
    "kaldi-fstext",
    "kaldi-nnet3",
    "kaldi-lm",
]
LIBRARY_DIRS = [KALDI_LIB_DIR]
MKL_ROOT = os.getenv("MKL_ROOT")
MKL_LIB_DIR = ""
if MKL_ROOT:
    MKL_LIB_DIR = os.path.join(MKL_ROOT, "lib")
    LIBRARY_DIRS.append(MKL_LIB_DIR)
    EXTRA_COMPILE_ARGS["cxx"] += ["-I{}/include".format(MKL_ROOT)]
    LIBRARIES += ["mkl_intel_lp64", "mkl_core", "mkl_sequential"]

LICENSE = "Apache 2.0"
VERSION = "1.0"

if PKWRAP_CPP_EXT == "no":
    setup(
        name=PACKAGE_NAME,
        version=VERSION,
        license=LICENSE,
        packages=find_packages(),
        scripts = [
            './satools/bin/anonymize',
            ],
    )
else:
    from torch.utils import cpp_extension
    setup(
        name=PACKAGE_NAME,
        version=VERSION,
        license=LICENSE,
        packages=find_packages(),
        scripts = [
            './satools/bin/anonymize',
            ],
        ext_modules=[
            cpp_extension.CppExtension(
                EXTENSION_NAME,
                SRC_FILES,
                language="c++",
                extra_compile_args=EXTRA_COMPILE_ARGS,
                libraries=LIBRARIES,
                library_dirs=LIBRARY_DIRS,
            )
        ],
        cmdclass={"build_ext": cpp_extension.BuildExtension.with_options(no_python_abi_suffix=True)},
    )
