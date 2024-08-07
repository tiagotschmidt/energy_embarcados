#!/bin/bash

# Array to store power usage values
power_usage=()


batch_sizes=(32 64 128)

# Get the current timestamp for folder naming
timestamp=$(date +%Y-%m-%d_%H-%M-%S)
for batch_size in "${batch_sizes[@]}"; do
  # Start monitoring GPU power usage in a separate process
  nvidia-smi --loop-ms=150 --format=csv --query-gpu=power.draw > "power_data_${batch_size}.log" &
  pid1=$!
  start_time=$((SECONDS))  # Capture start time in seconds

  # Run the python script 30 times with the current batch size
  for i in {1..1}; do
    python mlp_gpu.py --batch-size $batch_size
  done
  end_time=$((SECONDS))  # Capture end time in seconds
  elapsed_time=$((end_time - start_time))
  total_time=$((total_time + elapsed_time))  # Accumulate total time

  echo "Elapsed time: $elapsed_time seconds" > "${batch_size}_elapsed_time.txt"


  kill $pid1

  if [ ! -f "power_data_${batch_size}.log" ]; then
    echo "Error: power.log file not found."
    exit 1
  fi

  # Initialize variables
  total_power="0"
  count=0

  # Read the power values from the file
  while read line; do
    # Remove the ' W' part and convert to a floating-point number
    power=$(echo "$line" | cut -d' ' -f1)
    total_power=$(echo "$total_power + $power" | bc)
    count=$((count + 1))
  done < <(tail -n +2 "power_data_${batch_size}.log")

  # Calculate the average power
  average_power=$(echo "scale=2; $total_power / $count" | bc)

  # Print the result
  echo "Average power: $average_power W"
done
