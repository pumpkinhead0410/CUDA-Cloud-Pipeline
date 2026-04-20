#!/bin/bash

# latest update : 2026-04-20

# mandatory package download
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg
sudo apt-get install -y apt-transport-https ca-certificates curl gnupg-agent software-properties-common

# keyring directory setup (+ chmod)
sudo install -m 0755 -d /etc/apt/keyrings

# download Docker GPG key and save as keyring
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# GPG key check (optional)
# 1. 키 파일의 상세 정보 확인 (지문 및 발행자)
#    'gpg' 도구를 사용하여 개별 저장된 키링 파일의 지문을 출력합니다.
echo "--- Docker GPG Key Verification ---"
gpg --show-keys --fingerprint /etc/apt/keyrings/docker.gpg
# 2. 특정 지문(0EBF CD88)이 포함되어 있는지 자동으로 체크 (Optional)
if gpg --show-keys --fingerprint /etc/apt/keyrings/docker.gpg | grep -q "9DC8 5822 9FC7 DD3E 85AD  4433 490F 4105 6F19 5B98"; then
    echo "[Success] 공식 Docker Release Key(6F19 5B98)가 확인되었습니다."
else
    echo "[Warning] 키 지문이 일치하지 않습니다. 다운로드된 키를 확인하세요."
fi

# Repository key configuration (set gpg keyring)
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# add Docker repository
sudo add-apt-repository -y "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

sudo apt-get update
# sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Docker 엔진 및 CLI 확인
docker version

# 플러그인(Buildx, Compose) 확인
docker buildx version
docker compose version

# Active status check (optional)
echo "[Please check docker active status]"
echo -e "\033[43;31m[Press q or Ctrl + C if docker status is active]\033[0m"
sudo systemctl status docker
