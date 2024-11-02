#!/bin/bash

srun -J SoftMoE \
  -p a100q \
  -N 1 \
  -n 64 \
  --pty bash
