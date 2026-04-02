# CUDA-Cloud-Pipeline 🚀

> **MLOps system** that pushes hardware to the limit with CUDA/TensorRT and deploys it reliably on AWS using Kubernetes, Terraform, and GitHub Actions.

---

## Architecture Overview

```
GitHub Actions (CI/CD)
       │
       ▼
 Docker Multi-stage Build
 (CUDA builder → Python builder → Runtime)
       │
       ▼
 Amazon ECR  ──►  Amazon EKS (GPU node group)
                       │
              ┌────────┴────────┐
              │                 │
        Kubernetes HPA     TensorRT Engine
        (auto-scaling)     (GPU inference)
              │
        FastAPI Service
        (REST inference API)

Infrastructure managed by Terraform (IaC)
 ├── VPC (public + private subnets, NAT GW)
 ├── EKS cluster (CPU + GPU node groups)
 ├── ECR repository
 ├── NVIDIA device plugin (Helm)
 └── metrics-server (Helm, required for HPA)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **GPU Kernels** | C++ CUDA 12.4 (tiled GEMM, ReLU, Softmax) |
| **Python GPU Ops** | Python + Numba CUDA JIT |
| **Inference Engine** | TensorRT 10 (FP16 / INT8 optimisation) |
| **Model Conversion** | ONNX → TensorRT via `ModelConverter` |
| **Serving API** | FastAPI + Uvicorn |
| **Containerisation** | Docker multi-stage build (3 stages) |
| **Orchestration** | Kubernetes (EKS) with GPU scheduling |
| **Auto-scaling** | HorizontalPodAutoscaler (CPU + custom metrics) |
| **IaC** | Terraform – VPC, EKS, ECR modules |
| **CI/CD** | GitHub Actions (lint → test → build → deploy) |
| **Cloud** | AWS (EKS, ECR, EC2 g5.xlarge A10G GPU nodes) |

---

## Repository Structure

```
.
├── src/
│   ├── cuda/
│   │   ├── kernels/
│   │   │   ├── matrix_ops.cu      # Tiled GEMM + ReLU + Softmax (CUDA C++)
│   │   │   ├── matrix_ops.h
│   │   │   └── CMakeLists.txt     # CMake build (sm_75, sm_86)
│   │   ├── numba_ops.py           # Python Numba GPU kernels
│   │   └── __init__.py
│   ├── inference/
│   │   ├── trt_engine.py          # TensorRT engine wrapper + StubEngine
│   │   ├── model_converter.py     # PyTorch / ONNX → TRT conversion
│   │   └── __init__.py
│   └── api/
│       ├── main.py                # FastAPI app (lifespan, /healthz, /infer, /profile)
│       ├── schemas.py             # Pydantic request/response models
│       └── __init__.py
├── docker/
│   ├── Dockerfile                 # Multi-stage CUDA build
│   └── docker-compose.yml         # Local dev with GPU passthrough
├── kubernetes/
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── deployment.yaml            # GPU resource requests + security context
│   ├── service.yaml               # AWS NLB LoadBalancer
│   └── hpa.yaml                   # CPU + memory + custom-metric HPA
├── terraform/
│   ├── main.tf                    # Root module (VPC + EKS + ECR + Helm releases)
│   ├── variables.tf
│   ├── outputs.tf
│   ├── versions.tf                # Provider versions + S3 backend
│   └── modules/
│       ├── vpc/                   # VPC, subnets, IGW, NAT GW, route tables
│       └── eks/                   # EKS cluster, CPU/GPU node groups, Autoscaler IAM
├── .github/
│   └── workflows/
│       ├── ci.yml                 # Lint → Test → Docker build → Terraform validate
│       └── cd.yml                 # Build + push ECR → Deploy EKS → Terraform apply
├── tests/
│   ├── test_numba_ops.py          # NumPy-validated Numba kernel tests
│   └── test_api.py                # FastAPI integration tests (stub engine)
├── requirements.txt
├── requirements-dev.txt
├── setup.py
└── pyproject.toml                 # ruff + mypy + pytest config
```

---

## Quick Start

### Local development (CPU / stub mode)

```bash
# Install dependencies
pip install -r requirements-dev.txt

# Run tests (no GPU required)
USE_STUB=1 NUMBA_ENABLE_CUDASIM=1 pytest tests/ -v

# Start the API locally
USE_STUB=1 uvicorn src.api.main:app --host 0.0.0.0 --port 8080 --reload
```

### Docker (local GPU)

```bash
# Build & run (requires NVIDIA Container Toolkit)
docker compose -f docker/docker-compose.yml up --build
```

### Inference request example

```bash
curl -s -X POST http://localhost:8080/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs":       {"input": [0.1, 0.2, 0.3]},
    "input_shapes": {"input": [1, 3]}
  }' | python3 -m json.tool
```

---

## CUDA Kernels

### C++ CUDA (`src/cuda/kernels/matrix_ops.cu`)

- **Tiled GEMM** – 32×32 shared-memory tiles; configurable `alpha` / `beta` (SGEMM interface)
- **ReLU** – vectorised element-wise, 256-thread blocks
- **Softmax** – numerically stable (subtract row-max before exp)

Build the shared library:

```bash
cmake -S src/cuda/kernels -B build/cuda -DCMAKE_BUILD_TYPE=Release
cmake --build build/cuda --parallel
```

### Python Numba (`src/cuda/numba_ops.py`)

| Function | Description |
|---|---|
| `matrix_multiply(A, B)` | Tiled GEMM via `@cuda.jit` |
| `relu(data)` | Element-wise ReLU |
| `softmax(data)` | Row-wise numerically stable softmax |
| `layer_norm(data, gamma, beta)` | Per-sample layer normalisation |

---

## TensorRT Pipeline

```python
from src.inference.model_converter import ModelConverter
import torch

# 1. Convert PyTorch → ONNX → TRT engine
converter = ModelConverter.from_torch(
    model=my_resnet,
    example_inputs=torch.zeros(1, 3, 224, 224),
    engine_path="models/resnet50.engine",
    fp16=True,
    opt_shapes={"input": (1, 3, 224, 224)},
    min_shapes={"input": (1, 3, 224, 224)},
    max_shapes={"input": (8, 3, 224, 224)},
)
converter.build()

# 2. Run inference
from src.inference.trt_engine import TRTEngine
import numpy as np

with TRTEngine("models/resnet50.engine", enable_profiling=True) as engine:
    outputs = engine.infer({"input": np.random.rand(1, 3, 224, 224).astype("float32")})
    print(engine.profiling_summary())
```

---

## Infrastructure (Terraform)

```bash
cd terraform

# Configure backend (S3 + DynamoDB)
terraform init \
  -backend-config="bucket=my-tfstate-bucket" \
  -backend-config="key=cuda-cloud-pipeline/terraform.tfstate" \
  -backend-config="region=us-east-1" \
  -backend-config="dynamodb_table=terraform-locks"

# Preview changes
terraform plan

# Apply
terraform apply
```

Key resources created:
- **VPC** – 3-AZ with public/private subnets, NAT Gateways
- **EKS cluster** – Kubernetes 1.29, private API endpoint
- **CPU node group** – `m6i.large` (system workloads)
- **GPU node group** – `g5.xlarge` (NVIDIA A10G, inference)
- **ECR repository** – with lifecycle policy and KMS encryption
- **NVIDIA device plugin** – enables GPU scheduling in K8s
- **metrics-server** – required for HPA CPU/memory scaling

---

## CI/CD (GitHub Actions)

### CI (`ci.yml`) – every push / PR

1. **Lint** – `ruff check` + `ruff format --check` + `mypy`
2. **Test** – pytest with coverage upload to Codecov
3. **Docker build** – multi-stage build validation (no push)
4. **Terraform validate** – `terraform fmt -check` + `terraform validate`

### CD (`cd.yml`) – on push to `main`

1. **Build & push** – multi-stage CUDA image → Amazon ECR (OIDC auth)
2. **Deploy** – `kubectl apply` manifests + rolling update
3. **Smoke test** – `/healthz` endpoint check on live NLB
4. **Terraform apply** – infrastructure drift remediation

### Required GitHub Secrets

| Secret | Description |
|---|---|
| `AWS_DEPLOY_ROLE_ARN` | IAM role ARN for OIDC (GitHub → AWS) |
| `TF_STATE_BUCKET` | S3 bucket name for Terraform state |
| `TF_LOCK_TABLE` | DynamoDB table name for state locking |

---

## Auto-scaling Configuration

The `HorizontalPodAutoscaler` scales the inference deployment based on three signals:

| Metric | Target | Source |
|---|---|---|
| CPU utilisation | 70% | metrics-server |
| Memory utilisation | 80% | metrics-server |
| `inference_queue_depth` | avg 10 | Prometheus Adapter |

Scale-up: +2 pods / 60 s · Scale-down: -1 pod / 120 s (5-minute stabilisation window)

---

## Security Highlights

- Containers run as non-root (`uid=10001`)
- `readOnlyRootFilesystem: true` + `allowPrivilegeEscalation: false`
- All capabilities dropped
- ECR images scanned on push + KMS-encrypted
- EKS API endpoint: private access enabled, public access restricted
- AWS auth via OIDC (no long-lived access keys in CI)
- Terraform state encrypted at rest in S3
