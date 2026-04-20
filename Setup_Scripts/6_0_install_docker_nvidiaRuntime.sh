#!/bin/bash

# 1. 저장소 설정 (현재는 distribution 구분 없이 stable로 통합 권장)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update

# 2. nvidia-docker2 대신 toolkit 설치 (nvidia-docker2도 내부적으로 toolkit을 포함하지만, 직접 toolkit을 설치하는 것이 최신 표준)
sudo apt install -y nvidia-container-toolkit

# 3. Docker 런타임 설정 자동화 (이 부분이 핵심입니다!)
# 과거에는 daemon.json을 직접 수정했지만, 이제는 명령어로 설정 가능합니다.
sudo nvidia-ctk runtime configure --runtime=docker

sudo systemctl restart docker

# 1. 툴킷 설치 여부 확인
nvidia-ctk --version

# 2. GPU 연결 테스트 (가장 확실한 방법)
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi