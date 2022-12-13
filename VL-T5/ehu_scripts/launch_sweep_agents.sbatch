#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --job-name=vl-t5
#SBATCH --output=sweeps.out

export TRANSFORMERS_CACHE=/gscratch4/users/jcampos004/summer_internship/models/vlt5_output

source /gscratch4/users/jcampos004/summer_internship/venv/bin/activate


wandb login

CMD_OUTPUT=$(wandb sweep -p vlt5 sweeps_comet.yaml 2>&1)
SWEEP_ID=$(echo $CMD_OUTPUT | awk '{print $NF}')

echo $CMD_OUTPUT
echo $SWEEP_ID

## Run experiment
wandb agent $SWEEP_ID
