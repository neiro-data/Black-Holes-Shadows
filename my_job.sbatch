#!/bin/bash
#SBATCH --job-name=job_test
#SBATCH --output=/home/neiro/job_test%j.out
#SBATCH --error=/home/neiro/job_test%j.err
#SBATCH --mail-user=nelson.eiro@tecnico.ulisboa.pt
#SBATCH --mail-type=ALL
#SBATCH --time=1:00:00
#SBATCH --mem=4G

RUNPATH=/home/neiro/
cd $RUNPATH

./test_prog
