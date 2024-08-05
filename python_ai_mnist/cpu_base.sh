#!/bin/bash

# Define the batch sizes to test
batch_sizes=(32 64 128)

# Get the current timestamp for folder naming
timestamp=$(date +%Y-%m-%d_%H-%M-%S)

for batch_size in "${batch_sizes[@]}"; do
  # Create a folder with timestamp and batch size information
  mkdir "cpu_${batch_size}_${timestamp}"

  # Run the python script 30 times with the current batch size
  for i in {1..30}; do
    python cpu_base.py --batch-size $batch_size
    mv result_*.json "cpu_${batch_size}_${timestamp}/"
  done
done

echo "All runs completed!"
