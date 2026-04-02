variable "cluster_name" {
  description = "Name of the EKS cluster."
  type        = string
}

variable "cluster_version" {
  description = "Kubernetes version."
  type        = string
}

variable "vpc_id" {
  description = "VPC ID for the cluster."
  type        = string
}

variable "private_subnet_ids" {
  description = "Private subnet IDs for node groups."
  type        = list(string)
}

variable "cpu_node_instance_type" {
  description = "EC2 instance type for the CPU node group."
  type        = string
}

variable "gpu_node_instance_type" {
  description = "EC2 instance type for the GPU node group."
  type        = string
}

variable "gpu_node_min_size" {
  description = "Minimum GPU node count."
  type        = number
}

variable "gpu_node_max_size" {
  description = "Maximum GPU node count."
  type        = number
}

variable "gpu_node_desired_size" {
  description = "Desired GPU node count."
  type        = number
}

variable "project" {
  description = "Project tag."
  type        = string
}

variable "environment" {
  description = "Environment tag."
  type        = string
}
