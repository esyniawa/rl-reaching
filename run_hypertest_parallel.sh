#!/bin/bash

# Configuration
PARALLEL_JOBS=4                    # Number of parallel jobs
N_TRIALS=100                       # Number of trials per job
STUDY_NAME="small_model"           # Study name

# Record start time
START_TIME=$(date +%s)
echo "Starting parallel hyperparameter optimization at $(date)"
echo "Running $PARALLEL_JOBS parallel jobs with $N_TRIALS trials each"
echo "----------------------------------------"

PIDS=()
for ((i=0; i<$PARALLEL_JOBS; i++)); do
    SIM_ID=$((i + 1))
    echo "Starting job $((i+1))/$PARALLEL_JOBS with sim_id=$SIM_ID"

    python run_hypertest.py \
        --n_trials $N_TRIALS \
        --sim_id $SIM_ID \
        --study_name $STUDY_NAME &

    PIDS+=($!)
done

echo "All jobs launched. Waiting for completion..."

# Wait for all background jobs to complete
for pid in "${PIDS[@]}"; do
    wait $pid
    echo "Job with PID $pid completed"
done

# Calculate and display runtime
END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))

DAYS=$((RUNTIME / 86400))
HOURS=$(((RUNTIME % 86400) / 3600))
MINUTES=$(((RUNTIME % 3600) / 60))

echo "----------------------------------------"
echo "All jobs completed at $(date)"
echo "Total runtime: ${DAYS}:$(printf "%02d" $HOURS):$(printf "%02d" $MINUTES) (days:hours:minutes)"