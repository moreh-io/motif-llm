from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Motif inference using vLLM')
    parser.add_argument('--model-path', type=str, required=True, help="Directory path to load the model")
    parser.add_argument('--num-gpus', type=int, required=True, help="numbers of GPUs to use")

    return parser.parse_args()


def main(args):
    # Change tensor_parallel_size to GPU numbers you can afford
    model = LLM(args.model_path, tensor_parallel_size=args.num_gpus)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "유치원생에게 빅뱅 이론의 개념을 설명해보세요"},
    ]

    messages_batch = [tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=True, tokenize=False)]

    # vllm does not support generation_config of hf. So we have to set it like below
    sampling_params = SamplingParams(max_tokens=512, temperature=0, repetition_penalty=1.0, stop_token_ids=[tokenizer.eos_token_id])
    responses = model.generate(messages_batch, sampling_params=sampling_params)

    print(responses[0].outputs[0].text)


if __name__ == "__main__":
    args = parse_args()
    main(args)
