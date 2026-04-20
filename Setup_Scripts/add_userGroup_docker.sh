#!/bin/bash

# 1. 'docker' 그룹 생성 (이미 있을 수도 있지만 확인차)
sudo groupadd docker

# 2. 현재 로그인된 사용자($USER)를 docker 그룹에 추가
sudo usermod -aG docker $USER

# 3. 변경된 권한 적용 (현재 터미널 세션에 바로 반영)
newgrp docker