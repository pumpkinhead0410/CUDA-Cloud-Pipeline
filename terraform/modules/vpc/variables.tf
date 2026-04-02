variable "name" {
  description = "Name prefix for resources."
  type        = string
}

variable "cidr" {
  description = "VPC CIDR block."
  type        = string
}

variable "availability_zones" {
  description = "List of availability zones."
  type        = list(string)
}

variable "project" {
  description = "Project tag value."
  type        = string
}

variable "environment" {
  description = "Environment tag value."
  type        = string
}
