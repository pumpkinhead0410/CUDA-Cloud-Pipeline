provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# ── Data sources ──────────────────────────────────────────────────────────
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
  region     = data.aws_region.current.name
  name_prefix = "${var.project}-${var.environment}"
}

# ── VPC ───────────────────────────────────────────────────────────────────
module "vpc" {
  source = "./modules/vpc"

  name               = local.name_prefix
  cidr               = var.vpc_cidr
  availability_zones = var.availability_zones
  project            = var.project
  environment        = var.environment
}

# ── EKS cluster with GPU node groups ─────────────────────────────────────
module "eks" {
  source = "./modules/eks"

  cluster_name           = local.name_prefix
  cluster_version        = var.cluster_version
  vpc_id                 = module.vpc.vpc_id
  private_subnet_ids     = module.vpc.private_subnet_ids
  cpu_node_instance_type = var.cpu_node_instance_type
  gpu_node_instance_type = var.gpu_node_instance_type
  gpu_node_min_size      = var.gpu_node_min_size
  gpu_node_max_size      = var.gpu_node_max_size
  gpu_node_desired_size  = var.gpu_node_desired_size
  project                = var.project
  environment            = var.environment
}

# ── ECR repository ────────────────────────────────────────────────────────
resource "aws_ecr_repository" "app" {
  name                 = "${local.name_prefix}/inference-api"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "KMS"
  }
}

resource "aws_ecr_lifecycle_policy" "app" {
  repository = aws_ecr_repository.app.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 20 images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 20
        }
        action = { type = "expire" }
      }
    ]
  })
}

# ── Kubernetes providers (configured from EKS outputs) ───────────────────
provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_ca_certificate)
  token                  = module.eks.cluster_auth_token
}

provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_ca_certificate)
    token                  = module.eks.cluster_auth_token
  }
}

# ── NVIDIA device plugin (enables GPU scheduling in K8s) ─────────────────
resource "helm_release" "nvidia_device_plugin" {
  name       = "nvidia-device-plugin"
  repository = "https://nvidia.github.io/k8s-device-plugin"
  chart      = "nvidia-device-plugin"
  version    = "0.14.5"
  namespace  = "kube-system"

  set {
    name  = "failOnInitError"
    value = "false"
  }

  depends_on = [module.eks]
}

# ── metrics-server (required for HPA CPU/memory metrics) ─────────────────
resource "helm_release" "metrics_server" {
  name       = "metrics-server"
  repository = "https://kubernetes-sigs.github.io/metrics-server"
  chart      = "metrics-server"
  version    = "3.12.1"
  namespace  = "kube-system"

  depends_on = [module.eks]
}

# ── MLOps namespace ───────────────────────────────────────────────────────
resource "kubernetes_namespace" "mlops" {
  metadata {
    name = "mlops"
    labels = {
      "app.kubernetes.io/managed-by" = "Terraform"
      environment                    = var.environment
    }
  }

  depends_on = [module.eks]
}
