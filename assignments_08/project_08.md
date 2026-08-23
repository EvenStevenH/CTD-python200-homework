# Part 3: Project

## Part A: Supabase Setup

Supabase project setup was completed successfully.

<!-- ----------------------------------------------------------------------- -->

## Part B: Cloud Cost Analysis

Scenario A — Lightweight compute

- A t3.micro EC2 instance (1 vCPU, 1 GB RAM), on-demand pricing, running 8 hours per day, 5 days per week (approximately 160 hours per month).
- US East (N. Virginia) region.

Scenario B — Heavy analytics workload

- A p3.2xlarge EC2 instance (8 vCPU, 1 V100 GPU), running 24/7 for the full month (730 hours)
- an RDS db.m5.large instance (2 vCPU, 8 GB RAM)
- S3 Standard storage bucket with 1 TB of data.
- US East (N. Virginia) region

Scenario A costed 1.66 USD a month; this isn't surprising, given the low-end specifications used. Scenario B costed 2615.02 USD a month; this also wasn't surprising, given the high-end specifications and additional instances used. While exploring the calculator beyond the two required scenarios, it was interesting to see additional options you may need to consider, such as placeholder costs for licensing, data transfer (inbound/outbound), and monitoring services. In any case, the cost difference between the scenarios highlights that a GPU instance is worth its higher price when dealing with computationally intensive tasks

Video: https://youtu.be/dVNPy0iI0hY