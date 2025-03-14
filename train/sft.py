import os
from dataclasses import dataclass, field, asdict
from typing import Optional
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datasets import load_dataset, concatenate_datasets, DatasetDict
import transformers
import trl
import torch
from transformers import BitsAndBytesConfig

@dataclass
class TrainingConfig:
    model_name: str = field(default="Qwen/Qwen2.5-32B-Instruct")
    block_size: int = field(default=32768)
    wandb_project: Optional[str] = field(default="s1")
    wandb_entity: Optional[str] = field(default="hashimoto-group")
    train_file_path: Optional[str] = field(default='simplescaling/s1K_tokenized')
    dagger: bool = field(default=False)
    lora: bool = field(default=True)  # Enable LoRA by default
    quantization: Optional[str] = field(default=None)  # Options: None, "8bit", "4bit"

    def __post_init__(self):
        os.environ['WANDB_PROJECT'] = self.wandb_project
        os.environ['WANDB_ENTITY'] = self.wandb_entity

def train():
    # Parsing input arguments
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")

    # Set up model loading kwargs with torch_dtype set to float16
    load_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.float16,
    }
    
    # Apply quantization if specified
    if config.quantization == "8bit":
        load_kwargs["load_in_8bit"] = True
    elif config.quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"  # Alternatively "fp4" if preferred
        )
        load_kwargs["quantization_config"] = bnb_config

    # For 70B models, update additional kwargs
    if "70B" in config.model_name:
        load_kwargs.update({
            "attn_implementation": "flash_attention_2",
            "use_cache": False
        })
        
    # Load model with updated configurations
    model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name, **load_kwargs)
    
    # Apply LoRA if enabled
    if config.lora:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise ImportError("Please install peft to use LoRA: pip install peft")
        
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],  # Adjust based on the model architecture
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
    
    dataset = load_dataset(config.train_file_path)

    # Set up tokenizer and templates
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if "Llama" in config.model_name:
        instruction_template = "<|start_header_id|>user<|end_header_id|>"
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        tokenizer.pad_token = "<|reserved_special_token_5|>"  # Unused token for padding
    elif "Qwen" in config.model_name:
        instruction_template = "<|im_start|>user"
        response_template = "<|im_start|>assistant\n"
        tokenizer.pad_token = "<|fim_pad|>"

    # Data collator for training: only compute loss over assistant responses
    collator = trl.DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )
    args.dataset_text_field = 'text'
    args.max_seq_length = config.block_size
    trainer = trl.SFTTrainer(
        model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'] if 'test' in dataset else dataset['train'],
        args=args,
        data_collator=collator
    )

    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()

if __name__ == "__main__":
    train()
