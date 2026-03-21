#!/bin/bash

#SBATCH -J v100_E1

#SBATCH --nodelist=p-gc-3201
#SBATCH --partition=standard
#SBATCH --account=epo2_cr_default
#SBATCH --gres=gpu:v100:1


#SBATCH -o traj_error.out
#SBATCH -e traj_error.err
#SBATCH -N 1
#SBATCH -n 1

#SBATCH --mem=8G
#SBATCH -t 1:00:00

cd $SLURM_SUBMIT_DIR

echo "Running on node: $(hostname)"
echo "Allocated GPU(s): $CUDA_VISIBLE_DEVICES"
echo "GPU ID: ${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS}"
echo "GPU Information:"
nvidia-smi


python temp_quench.py -f control.cntrl

