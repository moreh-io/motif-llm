import os
import torch
import argparse
import random
import numpy

from tqdm import tqdm
from loguru import logger
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader

MODEL_PATH = "resources"
CONFIG_PATH = "resources/config.json"


def parse_args():
    parser = argparse.ArgumentParser(description='Motif training script for the MoAI platform (automatic parallelization applied)')

    parser.add_argument('--bfloat16',
                        type=bool,
                        default=True,
                        help="If set to false, the model will be trained using 32-bit floating-point precision.")
    parser.add_argument('--train-epochs', type=int, default=1, help="Number of training epochs")
    parser.add_argument('--log-interval', type=int, default=1, help="Number of steps between logging")
    parser.add_argument('--model-save-path', type=str, required=True, help="Directory path to save the model")
    parser.add_argument('--seed', type=int, default=1234, help='Random seed value')

    args = parser.parse_args()

    # Log the parsed arguments
    logger.info("Parsed arguments:")
    for arg in vars(args):
        logger.info(f"  {arg}: {getattr(args, arg)}")

    return args


def set_random_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)


class DataCollator:

    def __init__(self, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        texts = [item['text'] for item in batch]
        encoded_batch = self.tokenizer(texts,
                                       padding=True,
                                       truncation=True,
                                       max_length=self.max_length,
                                       return_tensors="pt")

        return encoded_batch


def main(args):
    """
    Entry point of the program for training Motif on the MoAI platform.
    
    The MoAI Platform performs advanced parallelization, which automatically optimizes and parallelizes based on the number of GPUs in use.
    For more information: https://docs.moreh.io/tutorials/ap_example/
    """
    torch.moreh.option.enable_advanced_parallelization(mixed_precision=args.bfloat16)
    set_random_seed(args.seed)
    model_name = "moreh/Llama-3-Motif-102B"

    dataset = load_dataset("flpelerin/tinystories-100k")['train']

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    collate_fn = DataCollator(tokenizer, max_length=4096)

    train_dataloader = DataLoader(dataset, batch_size=2048, shuffle=True, collate_fn=collate_fn)

    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_config(config).to('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)

    model.train()

    for train_epoch in range(args.train_epochs):
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            input_ids, attention_mask = batch['input_ids'].to('cuda'), batch['attention_mask'].to('cuda')
            outputs = model(input_ids=input_ids, labels=input_ids, attention_mask=attention_mask)
            loss = outputs[0]
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if step % args.log_interval == 0:
                logger.info(f"Step {step}: Loss {loss.item()}")

        # Save the trained model
        model.save_pretrained(os.path.join(args.model_save_path, str(train_epoch)))
        tokenizer.save_pretrained(os.path.join(args.model_save_path, str(train_epoch)))

        logger.info(f"Training epoch {train_epoch} completed")


if __name__ == "__main__":
    args = parse_args()
    main(args)
