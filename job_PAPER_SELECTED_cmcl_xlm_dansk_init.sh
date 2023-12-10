#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=DA_MLX
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=4

#activating the virtual environment
echo "Activating the virtual environment..."
source ${HOME}/anaconda3/etc/profile.d/conda.sh
conda activate cmcl

#running the actual code
echo "Starting the process..."
for lang in da # ru zh hi # nl not available on the hub
do
  for lrate in 0.001
  do
    for bsize in 8
    do
      CUDA_VISIBLE_DEVICES=0 python -u ${HOME}/cmcl_2022/process_data_XLM_DANSK_init.py -ln ${lang} -pm ffd_avg -lr ${lrate} -bs ${bsize} -epoch 50\
        &> ${HOME}/cmcl_2022/log_ffd_avg_XLM_init_${lang}_${lrate}_${bsize}_50 &
      CUDA_VISIBLE_DEVICES=1 python -u ${HOME}/cmcl_2022/process_data_XLM_DANSK_init.py -ln ${lang} -pm ffd_std -lr ${lrate} -bs ${bsize} -epoch 50\
        &> ${HOME}/cmcl_2022/log_ffd_std_XLM_init_${lang}_${lrate}_${bsize}_50 &
      CUDA_VISIBLE_DEVICES=2 python -u ${HOME}/cmcl_2022/process_data_XLM_DANSK_init.py -ln ${lang} -pm trt_avg -lr ${lrate} -bs ${bsize} -epoch 50\
        &> ${HOME}/cmcl_2022/log_trt_avg_XLM_init_${lang}_${lrate}_${bsize}_50 &
      CUDA_VISIBLE_DEVICES=3 python -u ${HOME}/cmcl_2022/process_data_XLM_DANSK_init.py -ln ${lang} -pm trt_std -lr ${lrate} -bs ${bsize} -epoch 50\
        &> ${HOME}/cmcl_2022/log_trt_std_XLM_init_${lang}_${lrate}_${bsize}_50
      wait
    done
  done
done