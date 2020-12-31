#based on https://github.com/tensorflow/tensorflow/issues/25939

ARG UBUNTU_VERSION=16.04

ARG CUDA=10.0
FROM nvidia/cuda:${CUDA}-base-ubuntu${UBUNTU_VERSION} as base
ARG CUDA=10.0
ARG CUDADASH=10-0
ARG CUDNN=7.4.1.5-1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-${CUDADASH} \
        cuda-cublas-${CUDADASH} \
        cuda-cufft-${CUDADASH} \
        cuda-curand-${CUDADASH} \
        cuda-cusolver-${CUDADASH} \
        cuda-cusparse-${CUDADASH} \
        curl \
        libcudnn7=${CUDNN}+cuda${CUDA} \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        pkg-config \
        software-properties-common \
        unzip \
        wget

ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

ARG PYTHON=python3.7

ENV LANG C.UTF-8

RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get install -y ${PYTHON} ${PYTHON}-dev

RUN wget https://bootstrap.pypa.io/get-pip.py
RUN ${PYTHON} get-pip.py
RUN ln -sf /usr/bin/${PYTHON} /usr/local/bin/python3
RUN ln -sf /usr/local/bin/pip /usr/local/bin/pip3

RUN pip3 --no-cache-dir install --upgrade \
    pip \
    setuptools

RUN ln -s $(which ${PYTHON}) /usr/local/bin/python

RUN apt-get update && apt-get install git cuda-cusparse-dev-10-0 cuda-cublas-dev-10-0  -y
RUN python -m pip install torch==1.2.0 torchvision==0.4.0 cython scikit-build mmcv==v0.6.1

ENV LD_LIBRARY_PATH=/usr/local/cuda:/usr/local/cuda/lib64

RUN apt-get update && apt-get install git -y
RUN git clone https://github.com/open-mmlab/mmskeleton.git
WORKDIR /mmskeleton
RUN python setup.py develop

# default runtime should be nvidia docker for compiling cython codes. 
WORKDIR /mmskeleton/mmskeleton/ops/nms
RUN python setup_linux.py develop

# installing mmdetection(v1.0rc1.zip)
WORKDIR /mmskeleton
RUN pip install -U pip
RUN python setup.py develop --mmdet
 
