# 🚀 Motif LLM
<p align="left">
        🤗 <a href="https://huggingface.co/collections/moreh/llama-3-motif-102b-collections-672b27de18685a38713b249c">Models on Hugging Face</a>&nbsp | <a href="https://moreh.io/blog/introducing-motif-a-high-performance-open-source-korean-llm-by-moreh-241202">Blog post</a> | <a href="#">(TBU) Tech paper</a>
</p>

Motif is a high-performance open-source Korean LLM with 102 billion parameters. It shows top-notch Korean language performance among open-source LLMs. This project demonstrates a simple example of how to run and train Motif. You can also directly chat with the instruct version on [Model Hub](https://model-hub.moreh.io).

## Benchmark scores

|Provider|Model|kmmlu_direct score|Source|
|---|---|---|---|
|Moreh|**Llama-3-Motif-102B (pretrained)**|**64.74**|Measured by Moreh
|Moreh|**Llama-3-Motif-102B-Instruct**|**64.81**|Measured by Moreh
|Meta|Llama-3-70B-Instruct|54.5|https://arxiv.org/abs/2402.11548
|Meta|Llama-3.1-70B-Instruct|52.1|https://arxiv.org/abs/2402.11548
|Meta|Llama-3.1-405B-Instruct|65.8|https://arxiv.org/abs/2402.11548
|Alibaba|Qwen2-72B-Instruct|64.1|https://arxiv.org/abs/2402.11548|https://arxiv.org/abs/2402.11548
|OpenAI|GPT-4-0125-preview|59.95|https://arxiv.org/abs/2402.11548
|OpenAI|GPT-4o-2024-05-13|64.11|Measured by Moreh
|Google|Gemini Pro|50.18|https://arxiv.org/abs/2402.11548
|LG|EXAONE 3.0|44.5|https://arxiv.org/pdf/2408.03541
|Naver|HyperCLOVA X|53.4|https://arxiv.org/abs/2402.11548
|Upstage|SOLAR-10.7B|41.65|https://arxiv.org/pdf/2404.01954

## 🖥️ System Requirements

To successfully run and train Motif, we recommend that the system meets the following requirements:

* Inference
  - **GPU**: Minimum of 4x NVIDIA A100-80GB (when using batch size ≤8, sequence length 8192, and bfloat16)
  - **Storage**: 1.3 TB of disk space for each checkpoint
  - **RAM**: 500 GB for model parameters
* Training
  - **GPU**: Minimum of 4x NVIDIA A100-80GB (when using batch size 2, sequence length 4096, and DeepSpeed O3)
  - **Storage**: 1.3 TB of disk space for each checkpoint
  - **RAM**: 2.5 TB for model parameters and optimizer state

## 🚀 Simple Usage

### Run with vLLM

Refer to this [link](https://github.com/vllm-project/vllm) to install vLLM.

```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Change tensor_parallel_size to GPU numbers you can afford
model = LLM("moreh/Llama-3-Motif-102B", tensor_parallel_size=4)
tokenizer = AutoTokenizer.from_pretrained("moreh/Llama-3-Motif-102B")
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "유치원생에게 빅뱅 이론의 개념을 설명해보세요"},
]

messages_batch = [tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=True, tokenize=False)]

# vLLM does not support generation_config of HF. So we have to set it like below
sampling_params = SamplingParams(max_tokens=512, temperature=0, repetition_penalty=1.0, stop_token_ids=[tokenizer.eos_token_id])
responses = model.generate(messages_batch, sampling_params=sampling_params)

print(responses[0].outputs[0].text)
```

### Run with Hugging Face Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "moreh/Llama-3-Motif-102B"

# all generation configs are set in generation_configs.json
model = AutoModelForCausalLM.from_pretrained(model_id).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_id)
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "유치원생에게 빅뱅 이론의 개념을 설명해보세요"},
]

messages_batch = tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=True, tokenize=False)
input_ids = tokenizer(messages_batch, padding=True, return_tensors='pt')['input_ids'].cuda()

outputs = model.generate(input_ids)
```

## Script Usage

This section introduces the installation and execution process on CUDA. For usage on the MoAI platform, please refer to the "Training on MoAI Platform" section.

### 📋 Prerequisites

- Python 3.8 or higher
- Docker

### 📦 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/moreh-io/motif-llm.git
   cd motif-llm
   ```

2. Start a Docker container:
   ```bash
   bash set_docker.sh
   ```

3. Install the required dependencies in the container:
   ```bash
   docker exec -it --user root motif_trainer /bin/bash
   cd ~/motif_trainer
   apt-get update
   apt-get install libaio-dev
   pip install -r requirements/cuda_requirements.txt
   ```

4. Set DeepSpeed's accelerate config:
   ```bash
    accelerate config
      In which compute environment are you running? This machine
      Which type of machine are you using? multi-GPU
      How many different machines will you use (use more than 1 for multi-node training)? [1]: 1
      Should distributed operations be checked while running for errors? This can avoid timeout issues but will be  slower. [yes/NO]: NO
      Do you wish to optimize your script with torch dynamo?[yes/NO]:NO
      Do you want to use DeepSpeed? [yes/NO]: yes
      Do you want to specify a json file to a DeepSpeed config? [yes/NO]: yes
      Please enter the path to the json DeepSpeed config file: resources/deepspeed_config.json
      Do you want to enable `deepspeed.zero.Init` when using ZeRO Stage-3 for constructing massive models? [yes/NO]: No
      Do you want to enable Mixture-of-Experts training (MoE)? [yes/NO]: no
      How many GPU(s) should be used for distributed training? [1]:8
      accelerate configuration saved at (user_hf_cache_path)
   ```

### 🔧 Usage

Run the sample training script with your desired model save path:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run.sh /path/to/save/model
```

Run the sample vLLM inference script with the saved model path:

```bash
bash inference.sh /path/to/save/model
```

## 🛠 Training on MoAI Platform

MoAI Platform enables training and fine-tuning of models with tens/hundreds of billions parameters on GPU clusters with minimal effort. To easily fine-tune the Motif model on a large number of GPUs, please follow the steps below.

1. Contact [Moreh](https://moreh.io/) for getting training environment
2. Install packages in ```requirements/moai_requirements.txt```
3. Run the ```example_train_model_moai.py``` script to use tens or hundreds of GPUs together
