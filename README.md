# CUDA-Cloud-Pipeline
Personal toy project that learning and improve my skills. 🫠

---

## 👽 Project Goal
1. 리소스 최적화(**CUDA/TensorRT**)
2. 클라우드 환경에서 안정적으로 배포 및 관리(**MLOps**)

## 🚙 Loadmap (by Steps)

### Step 1 : 표준화된 개발 환경 구축 (Foundation)

> 목표: 로컬과 클라우드 어디서든 동일하게 돌아가는 표준 만들기.

  1. : **Docker 기초 설계** - NVIDIA 공식 PyTorch 이미지를 베이스로 한 Dockerfile 작성.
  2. : **최적화 레이어 분리** - 모델 가중치를 제외한 런타임 환경만 담은 경량 이미지 빌드 테스트.
  3. : **개발 환경 자동화** - docker-compose를 이용해 로컬 GPU를 컨테이너 안으로 연결하는 스크립트 작성.
  4. : **docker push를 통해 Docker Hub나 GitHub Package에 첫 이미지 업로드.**

### Step 2 : LLM 최적화 엔진 (Core Tech)
> 목표: **Llama-3-8B**(예정) 를 **TensorRT-LLM**으로 가속화하기.

  1. : **Llama-3-8B** 모델 가중치 다운로드 및 로컬 로드 확인.
  2. : **TensorRT-LLM** 기본 빌드 - 제공되는 툴로 엔진(.engine) 파일 생성 시도.
  3. : **Nsight System** 분석 - 기본 추론 시 발생하는 병목 구간(Memory/Kernel) 리포트 작성.
  4. : **양자화(Quantization)** 적용 - FP16에서 INT4/FP8로 변환하며 성능과 정확도 비교.
  5. : 최적화 전/후 **Latency** 비교 차트를 GitHub에 기록.

### Step 3 . : 인프라 코드화 및 클라우드 준비 (Infrastructure)
> 목표 : **Terraform**으로 인프라 설계도 그리기 (without AWS)

  1. : **Terraform** 설치 및 **AWS CLI** 연동 (_with free-tier_).
  2. : **VPC 및 서브넷 설계** - 네트워크 기초 인프라를 .tf 파일로 작성.
  3. : **가성비 전략 수립** - Spot Instance 사용 옵션이 포함된 EC2 정의서 작성.
  4. : **IaC 검증** - terraform plan을 통해 실제로 어떤 자원이 생성될지 가상으로 확인.
  5. : **인프라 설계도**를 이미지로 그려서 GitHub에 업로드.

### Step 4 . : 배포 및 자동화 (MLOps)
> 목표 : 수동 작업을 없애는 자동 배포 파이프라인 완성.

  1. : **GitHub Actions 기초** - 코드 푸시 시 Docker 이미지가 자동 빌드되도록 설정.
  2. : **AWS EC2에 최적화 엔진 배포** - ssh를 이용한 자동 배포 스크립트 연동.
  3. : **Kubernetes(Kind/Minikube)** - 로컬에 K8s 클러스터 띄워보기.
  4. : **모니터링** - 로그 출력을 표준화하고 간단한 가동률 체크 스크립트 작성.
  5. : **전체 프로젝트 요약** : "성능 향상, 인프라 비용 절감" 수치 계산 및 업로드.
