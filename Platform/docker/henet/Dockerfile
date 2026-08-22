FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG HENET_REVISION=29ca81dd109cabe0a0c53ee354c4a74ad1559740
ARG DETECTRON2_REVISION=d1e04565d3bec8719335b88be9e9b961bf3ec464
ARG TORCH_CUDA_ARCH_LIST="8.6+PTX"

ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA=1
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        libgl1 \
        libglib2.0-0 \
        ninja-build \
        python3.10 \
        python3.10-dev \
        python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && python3.10 -m pip install --no-cache-dir --upgrade \
        pip==23.3.2 \
        setuptools==69.0.3 \
        wheel==0.41.2

RUN python3.10 -m pip install --no-cache-dir \
        torch==2.0.1+cu118 \
        torchvision==0.15.2+cu118 \
        torchaudio==2.0.2+cu118 \
        -f https://download.pytorch.org/whl/torch_stable.html \
    && python3.10 -m pip install --no-cache-dir \
        numpy==1.23.4 \
        mmcv-full==1.6.2 \
        -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html \
    && python3.10 -m pip install --no-cache-dir \
        mmdet==2.28.2 \
        mmsegmentation==0.30.0

RUN python3.10 -m pip install --no-cache-dir \
        aiobotocore==2.17.0 \
        boto3==1.35.93 \
        cloudpickle==2.2.1 \
        einops==0.7.0 \
        flytekit==1.13.15 \
        fsspec==2024.6.1 \
        fvcore==0.1.5.post20221221 \
        iopath==0.1.9 \
        kornia==0.7.1 \
        kubernetes==28.1.0 \
        lyft-dataset-sdk==0.0.8 \
        networkx==2.8.8 \
        numba==0.58.1 \
        numpy==1.23.4 \
        nuscenes-devkit==1.1.11 \
        opencv-python==4.8.1.78 \
        open3d==0.18.0 \
        pandas==1.4.4 \
        Pillow==9.5.0 \
        plyfile==1.0.2 \
        pyquaternion==0.9.9 \
        pyarrow==12.0.1 \
        refile==0.4.1 \
        s3fs==2024.6.1 \
        scikit-image==0.19.3 \
        scipy==1.10.1 \
        shapely==1.8.5 \
        tensorboard==2.14.0 \
        timm==0.9.12 \
        trimesh==2.35.39 \
        yacs==0.1.8

# The pinned checkout includes incomplete optional registries. Keep the
# registries required by the official HENet BEV segmentation config.
RUN git clone https://github.com/VDIGPKU/HENet.git /opt/HENet \
    && git -C /opt/HENet checkout "${HENET_REVISION}" \
    && test "$(git -C /opt/HENet rev-parse HEAD)" = "${HENET_REVISION}" \
    && git clone --depth 1 --branch v0.6 \
        https://github.com/facebookresearch/detectron2.git \
        /tmp/detectron2 \
    && test "$(git -C /tmp/detectron2 rev-parse HEAD)" = "${DETECTRON2_REVISION}" \
    && cp -a /tmp/detectron2/detectron2/data \
        /opt/HENet/detr2/detectron2/data \
    && cp -a /tmp/detectron2/detectron2/utils/events.py \
        /opt/HENet/detr2/detectron2/utils/events.py \
    && rm -rf /tmp/detectron2 \
    && printf '%s\n' \
        'class GlobalConfig:' \
        '    nvo = None' \
        '    signature = ""' \
        '    enable_cache = False' \
        '' \
        'class GlobalBEVCache:' \
        '    force_initialize = False' \
        '    forward_ind = 0' \
        > /opt/HENet/tools/watermark_cache.py \
    && printf '%s\n' \
        'from mmdet.datasets.builder import build_dataloader' \
        'from .builder import DATASETS, PIPELINES, build_dataset' \
        > /opt/HENet/mmdet3d/datasets/__init__.py \
    && printf '%s\n' \
        'from .backbones import *  # noqa: F401,F403' \
        'from .builder import (BACKBONES, DETECTORS, FUSION_LAYERS, HEADS, LOSSES, MIDDLE_ENCODERS, NECKS, ROI_EXTRACTORS, SEGMENTORS, SHARED_HEADS, VOXEL_ENCODERS, MODELS, ALGORITHM, build_backbone, build_detector, build_fusion_layer, build_head, build_loss, build_middle_encoder, build_model, build_neck, build_roi_extractor, build_shared_head, build_voxel_encoder, build_algorithm)' \
        'from .dense_heads import *  # noqa: F401,F403' \
        'from .detectors.bevdet import BEVDepth4D' \
        'from .detectors.bevdet_mix import BEVStereo4D_mix_encoder' \
        'from .losses import *  # noqa: F401,F403' \
        'from .necks import *  # noqa: F401,F403' \
        > /opt/HENet/mmdet3d/models/__init__.py \
    && printf '%s\n' \
        'from .base import Base3DDetector' \
        'from .bevdet import BEVDepth4D' \
        'from .bevdet_mix import BEVStereo4D_mix_encoder' \
        > /opt/HENet/mmdet3d/models/detectors/__init__.py \
    && sed -i \
        '/from .petr_transformer import \*/d' \
        /opt/HENet/mmdet3d/models/utils/__init__.py \
    && sed -i \
        '/from .cb_bevdet import CBBEVStereo4D/d' \
        /opt/HENet/mmdet3d/models/detectors/bevdet_mix.py \
    && sed -i \
        's/if torch.cuda.is_available() and CUDA_HOME is not None:/if (torch.cuda.is_available() or os.environ.get("FORCE_CUDA") == "1") and CUDA_HOME is not None:/' \
        /opt/HENet/mmdet3d/ops/deformattn/setup.py \
    && cd /opt/HENet/mmdet3d/ops/csrc \
    && python3.10 setup.py build_ext --inplace \
    && cd /opt/HENet/mmdet3d/ops/deformattn \
    && python3.10 setup.py build install \
    && cd /opt/HENet \
    && python3.10 setup.py develop \
    && cd /opt/HENet/detr2 \
    && python3.10 setup.py develop

WORKDIR /workspace

COPY Model/ /workspace/Model/
COPY Platform/pipelines/ /workspace/Platform/pipelines/

ENV PYTHONPATH=/opt/HENet:/workspace/Model:/workspace

RUN python3.10 -m pip install --no-cache-dir \
        black==21.4b2 \
        fairscale==0.4.13 \
        ipython==8.12.2 \
    && python3.10 -m pip check \
    && python3.10 -c \
    "import torch; \
from Platform.pipelines import henet_occupancy, henet_runtime, workflows; \
assert henet_occupancy.HENET_REVISION; \
assert henet_runtime.HENET_INPUT_WIDTH == 1152; \
assert workflows.precompute_henet_occupancy_artifacts.name; \
assert torch.__version__.startswith('2.0.1')"
