variable "aws_region" {
  description = "AWS region to deploy resources in."
  type        = string
  default     = "us-east-1"
}

variable "project" {
  description = "Short project identifier used in resource names and tags."
  type        = string
  default     = "cuda-pipeline"
}

variable "environment" {
  description = "Deployment environment (e.g. dev, staging, production)."
  type        = string
  default     = "production"
}

variable "cluster_version" {
  description = "Kubernetes version for the EKS cluster."
  type        = string
  default     = "1.29"
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC."
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "List of availability zones (must be ≥ 2)."
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b", "us-east-1c"]
}

variable "cpu_node_instance_type" {
  description = "EC2 instance type for the general-purpose (CPU) node group."
  type        = string
  default     = "m6i.large"
}

variable "gpu_node_instance_type" {
  description = "EC2 instance type for the GPU-accelerated node group."
  type        = string
  # g5.xlarge – NVIDIA A10G, 24 GB VRAM
  default = "g5.xlarge"
}

variable "gpu_node_min_size" {
  description = "Minimum number of GPU nodes."
  type        = number
  default     = 1
}

variable "gpu_node_max_size" {
  description = "Maximum number of GPU nodes (HPA ceiling)."
  type        = number
  default     = 10
}

variable "gpu_node_desired_size" {
  description = "Desired / initial number of GPU nodes."
  type        = number
  default     = 2
}

variable "ecr_image_tag" {
  description = "Container image tag to deploy (typically the Git SHA)."
  type        = string
  default     = "latest"
}
