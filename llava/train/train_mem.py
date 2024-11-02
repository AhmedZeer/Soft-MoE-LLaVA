import logging
from llava.train.train import train

if __name__ == "__main__":
    logging.info("0-"*10 + " Soft-MoE-LLaVA. " + "0-"*10)
    train(attn_implementation="flash_attention_2")
