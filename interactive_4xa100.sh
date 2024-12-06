#!/bin/bash

srun -J DoubleJuice \
  -p a100x4q \
  -N 2 \
  -n 64 \
  --pty bash
