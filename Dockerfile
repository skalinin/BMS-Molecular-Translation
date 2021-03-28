FROM nvidia/cuda:11.2.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update &&\
    apt-get -y install \
    build-essential yasm nasm cmake \
    unzip git wget curl nano tmux \
    sysstat libtcmalloc-minimal4 pkgconf autoconf libtool flex bison \
    python3 python3-pip python3-dev python3-setuptools \
    libglib2.0-0 libgl1-mesa-glx \
    libsm6 libxext6 libxrender1 libssl-dev libx264-dev libsndfile1 libmp3lame-dev &&\
    ln -s /usr/bin/python3 /usr/bin/python &&\
    ln -s /usr/bin/pip3 /usr/bin/pip &&\
    apt-get clean &&\
    apt-get autoremove &&\
    rm -rf /var/lib/apt/lists/* &&\
    rm -rf /var/cache/apt/archives/*

# Build nvidia codec headers
RUN git clone -b sdk/9.1 --single-branch https://git.videolan.org/git/ffmpeg/nv-codec-headers.git &&\
    cd nv-codec-headers && make install &&\
    cd .. && rm -rf nv-codec-headers

# Build ffmpeg with nvenc support
RUN git clone --depth 1 -b release/4.3 --single-branch https://github.com/FFmpeg/FFmpeg.git &&\
    cd FFmpeg &&\
    mkdir ffmpeg_build && cd ffmpeg_build &&\
    ../configure \
    --enable-cuda \
    --enable-cuvid \
    --enable-shared \
    --disable-static \
    --disable-doc \
    --extra-cflags=-I/usr/local/cuda/include \
    --extra-ldflags=-L/usr/local/cuda/lib64 \
    --enable-gpl \
    --enable-libmp3lame \
    --extra-libs=-lpthread \
    --nvccflags="-gencode arch=compute_75,code=sm_75" &&\
    make -j$(nproc) && make install && ldconfig &&\
    cd ../.. && rm -rf FFmpeg

# Upgrade pip for cv package instalation
RUN pip3 install --upgrade pip==21.0.1

RUN pip3 install --no-cache-dir numpy==1.19.5

# Install PyTorch
RUN pip3 install --no-cache-dir \
    torch==1.8.0+cu111 \
    torchvision==0.9.0+cu111 \
    torchaudio==0.8.0 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Install python ML packages
RUN pip3 install --no-cache-dir \
    opencv-python==4.5.1.48 \
    pandas==1.2.3 \
    pudb==2020.1 \
    scikit-learn==0.24.1 \
    scipy==1.6.1 \
    timm==0.4.5 \
    python-Levenshtein==0.12.2 \
    ipywidgets==7.6.3 \
    albumentations==0.5.2 \
    requests==2.25.1 \
    pytorch-lightning==1.2.4 \
    fonttools==4.21.1 \
    matplotlib==3.3.4 \
    notebook==6.2.0 \
    Pillow==8.1.2

ENV PYTHONPATH $PYTHONPATH:/workdir/src
ENV TORCH_HOME=/workdir/data/.torch

WORKDIR /workdir
