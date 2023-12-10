#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=DE_MLX
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=4

#activating the virtual environment
echo "Activating the virtual environment..."
source ${HOME}/anaconda3/etc/profile.d/conda.sh
conda activate cmcl

#running the actual code
echo "Starting the process..."
for lang in de # ru zh hi # nl not available on the hub
do
  for seed in 42
  do
    CUDA_VISIBLE_DEVICES=0 python -u ${HOME}/cmcl_2022/process_data_XLM_MULTILING_stack.py -seed ${seed} -ln ${lang} -pm ffd_avg -lr 0.001 -bs 8 -epoch 50\
      &> ${HOME}/cmcl_2022/log_ffd_avg_XLM_stack_${seed}_${lang}_0.001_8_50 &
    CUDA_VISIBLE_DEVICES=1 python -u ${HOME}/cmcl_2022/process_data_XLM_MULTILING_stack.py -seed ${seed} -ln ${lang} -pm ffd_std -lr 0.001 -bs 8 -epoch 50\
      &> ${HOME}/cmcl_2022/log_ffd_std_XLM_stack_${seed}_${lang}_0.001_8_50 &
    CUDA_VISIBLE_DEVICES=2 python -u ${HOME}/cmcl_2022/process_data_XLM_MULTILING_stack.py -seed ${seed} -ln ${lang} -pm trt_avg -lr 0.001 -bs 8 -epoch 50\
      &> ${HOME}/cmcl_2022/log_trt_avg_XLM_stack_${seed}_${lang}_0.001_8_50 &
    CUDA_VISIBLE_DEVICES=3 python -u ${HOME}/cmcl_2022/process_data_XLM_MULTILING_stack.py -seed ${seed} -ln ${lang} -pm trt_std -lr 0.001 -bs 8 -epoch 50\
      &> ${HOME}/cmcl_2022/log_trt_std_XLM_stack_${seed}_${lang}_0.001_8_50
    wait
  done
done