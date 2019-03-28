#!/bin/bash

PS=1  # Parameter servers
N_WORKER=5  # Parallel workers
N_DROPPED=1  # Dropped gradients
N_AGG=$((${N_WORKER}-${N_DROPPED}))  # Gradients to aggregate

DATETIME=`date +'%Y%m%d-%H%M%S'`

p=0
while [ ${p} -lt ${PS} ]
do
    echo "Starting Parameter Server $p"
    CUDA_VISIBLE_DEVICES='' python3 -m main_dist.py --job_name="ps" --task_index=${p} --agg=${N_AGG} &>/dev/null &
    p=$(( $p + 1 ))
done

w=0
while [ ${w} -lt ${N_WORKER} ]
do
    echo "Starting Worker $w"
    CUDA_VISIBLE_DEVICES='' python3 -m main_dist.py --job_name="worker" --task_index=${w} --agg=${N_AGG} &>/dev/null &
    w=$(( $w + 1 ))
done