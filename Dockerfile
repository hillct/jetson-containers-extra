ARG BASE_IMAGE=nvcr.io/nvidia/l4t-base:35.3.1
FROM ${BASE_IMAGE} as base
ARG DEBIAN_FRONTEND=noninteractive
ARG sm=87
ARG USE_DISTRIBUTED=1                    # skip setting this if you want to enable OpenMPI backend
ARG USE_QNNPACK=0
ARG CUDA=11-8
# nvidia-l4t-core is a dependency for the rest
# of the packages, and is designed to be installed directly
# on the target device. This because it parses /proc/device-tree
# in the deb's .preinst script. Looks like we can bypass it though:
RUN \
    echo "deb https://repo.download.nvidia.com/jetson/common r35.3 main" >> /etc/apt/sources.list && \
    echo "deb https://repo.download.nvidia.com/jetson/t194 r35.3 main" >> /etc/apt/sources.list && \
    apt-key adv --fetch-key http://repo.download.nvidia.com/jetson/jetson-ota-public.asc && \
    mkdir -p /opt/nvidia/l4t-packages/ && \
    touch /opt/nvidia/l4t-packages/.nv-l4t-disable-boot-fw-update-in-preinstall && \
    rm -f /etc/ld.so.conf.d/nvidia-tegra.conf && apt-get update && \
    apt-get install -y --no-install-recommends nvidia-l4t-core && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/arm64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && apt-get update && apt-get install -y --no-install-recommends cuda-${CUDA} && \
    apt-get -y upgrade &&  apt-get clean && rm -rf /var/lib/apt/lists/* cuda-keyring_1.0-1_all.deb

FROM base as builder

ARG dist_dir=/dist
ENV DIST_DIR=${dist_dir}
RUN    apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends software-properties-common build-essential \
        autoconf automake cmake libb64-dev libre2-dev libssl-dev libtool libboost-dev libcurl4-openssl-dev rapidjson-dev zlib1g-dev patchelf \
        libopenblas-dev libopenmpi-dev git nano python-is-python3  python3-pip \
        && apt-get clean && rm -rf /var/lib/apt/lists/* && mkdir -p ${DIST_DIR}

FROM builder as builder-nccl

ARG DEBIAN_FRONTEND=noninteractive
ARG sm=87

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends devscripts dh-make && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/NVIDIA/nccl.git && cd nccl && \
	make -j"$(grep -c ^processor /proc/cpuinfo)" src.build NVCC_GENCODE="-gencode=arch=compute_${sm},code=sm_${sm}" && \
	make pkg.debian.build 

FROM builder as builder-pytorch
ARG DEBIAN_FRONTEND=noninteractive
ARG sm=87
ARG USE_NCCL=1
ARG USE_PYTORCH_QNNPACK=0
ENV CMAKE_CUDA_ARCHITECTURES=${sm} \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
ENV PATH=${CUDA_HOME}/bin:${PATH}

WORKDIR /workspace
COPY --from=builder-nccl /workspace/nccl/build/pkg/deb/libnccl* /dist/
RUN apt-get update && apt-get install -y --no-install-recommends pkg-config python3-dev  && apt-get install /dist/*.deb && \
	apt-get install -y --no-install-recommends g++ devscripts debhelper fakeroot && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN git clone --recursive --branch v2.1.0 https://github.com/pytorch/pytorch/ && cd pytorch && \
    pip3 install -r requirements.txt && pip3 install -U scikit-build ninja cmake==3.24.3 && \
    pip3 install --upgrade pip setuptools wheel && rm -rf /root/.cache && \
    TORCH_CUDA_ARCH_LIST=$(echo "scale=1; $sm / 10" | bc) python3 setup.py bdist_wheel
# Moved devscrips package installation after pip installation, due to conflicts - fixed (probably) by making nccl a staged build element
# The TORCH_CUDA_ARCH_LIST BS needs to be dealt wih upsream in pyorch source/build and should use the $sm integer notation rather than floaing point

FROM builder as jupyter
COPY --from=builder-nccl /workspace/nccl/build/pkg/deb/libnccl* /dist/
COPY --from=builder-pytorch /workspace/pytorch/dist/* /dist/

RUN apt-get update && apt-get install -y --no-install-recommends pkg-config python3-dev  && apt-get install /dist/*.deb && \
    pip3 install /dist/*.whl && pip3 install -U protobuf numpy ipywidgets fastapi gradio requests>==2.31 jupyterlab && apt-get clean && rm -rf /root/.cache /var/lib/apt/lists/*
WORKDIR /workspace
#ENTRYPOINT ["jupyter-lab","--allow-root","--ip=*","--no-browser"]
ENTRYPOINT ["/usr/bin/bash"]
