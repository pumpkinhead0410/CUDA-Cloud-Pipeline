# 1단계: 베이스 이미지 설정 (NVIDIA CUDA 환경)
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# 2단계: 메타데이터 설정 (준호 님의 '매뉴얼 v3.1' 정신 반영)
LABEL maintainer="Junho Kim <github.com/pumpkinhead0410>"
LABEL description="Optimized LLM Inference Environment for CUDA-Cloud-Pipeline"

# 3단계: 시스템 업데이트 및 필수 라이브러리 설치 (레이어 최적화)
# '&& \'를 사용하여 레이어 수를 줄이고, 캐시 효율을 높입니다.
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4단계: 작업 디렉토리 설정
WORKDIR /app

# 5단계: Python 환경 구축
# 자주 변경되는 소스 코드보다 패키지 설치를 먼저 두어 빌드 속도를 최적화합니다.
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 6단계: 기본 실행 명령어 (생략 가능, 추후 최적화 엔진 실행 시 설정)
CMD ["/bin/bash"]