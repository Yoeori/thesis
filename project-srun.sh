#!/usr/bin/env bash
#SBATCH -p ipuq
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 1-1:00 # time (D-HH:MM)
#SBATCH -o /home/yoeri/output/thesis-ipu/slurm.%N.%j.out
#SBATCH -e /home/yoeri/output/thesis-ipu/slurm.%N.%j.err

FILES=(~/data/tests/*.mtx)    
FILE=${FILES[$SLURM_ARRAY_TASK_ID]}

ulimit -s 10240
mkdir -p /home/yoeri/output/thesis-ipu

module purge
module load slurm/20.02.7
module load graphcore/vipu/1.18.0
module load graphcore/sdk/3.0.0

# TODO: Setup build system? Make sure build is set to release (-O3)?
export IPUOF_CONFIG_PATH=/cm/shared/apps/graphcore/vipu/etc/ipuof.conf.d/p64_cl_a01_a16.conf
cd /home/yoeri/thesis/build
srun ./matrix-ipu-calc $FILE -e spmv --seed 1 --rounds 10
