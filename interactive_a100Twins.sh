#!/bin/bash

srun -J SoftMoE \
  -p a100q \
  -N 2 \
  -n 64 \
  --pty bash
