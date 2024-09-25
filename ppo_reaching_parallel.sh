#!/bin/bash

parallel=$1
durchgaenge=$2

# Function to convert seconds to days:hours:minutes format
seconds_to_dhm() {
    local seconds=$1
    local days=$((seconds / 86400))
    local hours=$(( (seconds % 86400) / 3600 ))
    local minutes=$(( (seconds % 3600) / 60 ))
    printf "%d:%02d:%02d" $days $hours $minutes
}

# Create or clear the output files
parallel_runtime_file="parallel_runtimes_ppo.txt"
total_runtime_file="total_runtime_ppo.txt"
> "$parallel_runtime_file"
> "$total_runtime_file"

# Start timing the entire script
start_time=$(date +%s)

# start paradigma
for durchgang in $(seq $durchgaenge); do
    parallel_start_time=$(date +%s)

    for i in $(seq $parallel); do
        let y=$i+$parallel*$((durchgang - 1))
        python run_reaching_ppo.py --id $y --save 1 &
    done
    wait

    parallel_end_time=$(date +%s)
    parallel_duration=$((parallel_end_time - parallel_start_time))
    parallel_runtime=$(seconds_to_dhm $parallel_duration)

    # Calculate the SimID range for this parallel run
    start_sim_id=$((1 + parallel * (durchgang - 1)))
    end_sim_id=$((parallel * durchgang))

    echo "Parallel run $durchgang (SimID $start_sim_id-$end_sim_id) completed in $parallel_runtime"
    echo "SimID $start_sim_id-$end_sim_id: $parallel_runtime" >> "$parallel_runtime_file"

    sleep 5
done

# End timing the entire script
end_time=$(date +%s)
total_duration=$((end_time - start_time))
total_runtime=$(seconds_to_dhm $total_duration)

echo "Total runtime: $total_runtime"
echo "Total runtime: $total_runtime" >> "$total_runtime_file"
