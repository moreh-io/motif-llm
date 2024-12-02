docker pull huggingface/accelerate:gpu-deepspeed-release-1.0.1
apt-get install -y nvidia-container-toolkit

docker run -it --gpus all --name motif_trainer -d \
           --network=host \
           --ipc=host \
           -v $PWD:/root/motif_trainer/ \
           huggingface/accelerate:gpu-deepspeed-release-1.0.1 \
           sleep infinity
