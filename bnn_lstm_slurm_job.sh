#!/bin/bash
#SBATCH -J BNN_correction-bio-berry                # name of the job
#SBATCH -p shared                # name of the partition: available options "standard,standard-low,gpu,gpu-low,hm"
#SBATCH -n 16                    # no of processes or tasks
#SBATCH -t 3-00:00:00                # walltime in HH:MM:SS, Max value 72:00:00 #! 
#SBATCH --mem-per-cpu=64G
#SBATCH --mail-user=subhadeepiitkgpcoral@gmail.com        # user's email ID where job status info will be sent
#SBATCH --mail-type=ALL        # Send Mail for all type of event regarding the job

export PATH=/scratch/20cl91p02/anaconda3/bin:$PATH
# Navigate to the directory where your script is located
cd /scratch/20cl91p02/ANN_BIO/BNN/

python Quota-based-prediction-BNN+LSTM.py