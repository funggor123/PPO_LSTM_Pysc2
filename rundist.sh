#!/bin/bash

read -p "[Track] Enter the task type (worker/ps) " job
read -p "[Track] Enter task_index (0-10) " index
read -p "[Track] Enter a gradient " agg
echo "[Track] Task Type = $job , Task Index = $index , A Gradient = $agg"
python3 main_dist.py --job_name=$job --task_index=$index --agg=$agg
