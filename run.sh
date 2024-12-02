export TOKENIZERS_PARALLELISM=false

accelerate launch example_train_model_cuda.py --model-save-path $1
