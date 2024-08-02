#!/bin/bash

# Array to store power usage values
power_usage=()

# Start monitoring GPU power usage in a separate process
nvidia-smi --loop-ms=150 --format=csv --query-gpu=power.draw > power_data.log &
pid1=$!
# Loop for 100 iterations
for ((i=1; i<=100; i++)); do
    echo "Iteration $i: Running small_haar"
    ./small_haar 
done

kill $pid1

if [ ! -f power_data.log ]; then
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
  echo $power
  echo "$total_power + $power"
  total_power=$(echo "$total_power + $power" | bc)
  count=$((count + 1))
done < <(tail -n +2 power_data.log)

# Calculate the average power
average_power=$(echo "scale=2; $total_power / $count" | bc)

# Print the result
echo "Average power: $average_power W"
