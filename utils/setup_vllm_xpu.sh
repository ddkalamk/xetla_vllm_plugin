#!/bin/bash -x

# https://docs.vllm.ai/en/v0.19.0/getting_started/installation/gpu/#intel-xpu
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate

PT_XETLA_DIR=`realpath .`

#conda install -y ninja setuptools tqdm future cmake numpy pyyaml scikit-learn pydot -c conda-forge
#conda install -y gperftools -c conda-forge
#conda install -y pybind11 -c conda-forge

git clone -b xetla_v0.19.0 https://github.com/ddkalamk/vllm.git vllm
cd vllm
pip install --upgrade pip
# pip install "cmake>=3.26.1" wheel packaging ninja "setuptools-scm>=8" numpy
pip install -v -r requirements/xpu.txt

# VLLM_TARGET_DEVICE=xpu python setup.py install
VLLM_TARGET_DEVICE=xpu pip install --no-build-isolation -e . -v

pip uninstall -y triton triton-xpu
pip install triton-xpu==3.7.0 --extra-index-url https://download.pytorch.org/whl/xpu

cd $PT_XETLA_DIR
python setup.py install

