#!/bin/bash
# EC2 machine setup for running mistral as an LLM server
# March 17, 2024
#
# I booted a g4dn.xlarge instance with the latest Nvidia deep learning pytorch AMI:
# https://us-east-1.console.aws.amazon.com/ec2/home?region=us-east-1#Images:visibility=public-images;imageId=ami-09f1e96e440a7ba0a
#
# Note: don't do the similarly named "base" one -- that one is missing a bunch of stuff.
#
# I set it to have a root magnetic HDD of 200GB, which I figure will be enough. It's the cheapest GPU
# machine, at $12.624 per day, which seems okay for now. It should have 16GB of regular and video
# RAM, and 4 vCPUs.
#
# It turns out there's an additional nvme storage of 125GB, so I don't need to make the root so big.

# list conda envs
conda env list

# there's just base and pytorch, so i enable the latter
conda activate pytorch
conda init bash

# log out, log back in for changes to take effect
# i also copy over some key lines from my .bashrc to make it easier to work
# i also add my huggingface token as env variable HF_TOKEN
conda activate pytorch

# remove crap in the home dir
rm -i *I* O*

# i follow the guide for mistral vllm here: https://docs.mistral.ai/self-deployment/vllm/
# first step (note the addition of --dtype half as per the guide)
docker run --gpus all \
    -e HF_TOKEN=$HF_TOKEN -p 8000:8000 \
    ghcr.io/mistralai/mistral-src/vllm:latest \
    --host 0.0.0.0 \
    --model mistralai/Mistral-7B-Instruct-v0.2
    --dtype half

# it downloads a bunch of stuff in parallel, taking about 10 mins total
# but then it errors due to OOM:
# torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 384.00 MiB. GPU 0 has a total capacty of 14.58 GiB of which 311.56 MiB is free. Process 31047 has 14.27 GiB memory in use. Of the allocated memory 14.00 GiB is allocated by PyTorch, and 12.74 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF

# I have to run so i stop the instance (also to see if i can do that and restart it fine)

# ok back
# DO NOT pip install vllm, it will trash the torch and other libs!

# i found various params that might help:
#   --gpu-memory-utilization which defaults to 0.9
#   --quantization (-q) {awq,squeezellm,None}
#     Method used to quantize the weights.
#   --max-parallel-loading-workers <workers>
#     Load model sequentially in multiple batches, to avoid RAM OOM when using tensor parallel and large models.
#   --swap-space <size>
#     CPU swap space size (GiB) per GPU.
#
# so I first tried setting the gpu mem util to 1.0:
#   watching nvidia-smi, it looks like we have a full gb to spare while it's loading: 14143/15360,
#   whereas if we did apply the 0.9 limit, then that would be 13824, which would indeed cause OOM.
#   however, we still have the exact same error
#
# next I tried also setting quantization to awq:
#   it says "cannot find the config file for awq.
#   I try it with squeezellm and it says the same.
#
# I decided to just try running my own llm.py script.
# first i needed some libs:
pip install accelerate
#
