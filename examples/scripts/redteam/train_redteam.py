#!/usr/bin/env python
import logging

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser

from trl import ModelConfig, ScriptArguments
from trl.trainer.redteam_ppo_config import RedTeamPPOConfig
from trl.trainer.redteam_ppo_trainer import RedTeamPPOTrainer


# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # Parse arguments
    parser = HfArgumentParser((ScriptArguments, RedTeamPPOConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_into_dataclasses()

    logger.info(f"Script arguments: {script_args}")
    logger.info(f"Training arguments: {training_args}")
    logger.info(f"Model config: {model_config}")

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load tokenizer
    logger.info(f"Loading tokenizer from {model_config.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")

    logger.info(f"Loading policy model from {model_config.model_name_or_path}")
    policy = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16 if training_args.fp16 else torch.float32,
    )

    logger.info("Loading reference policy model")
    ref_policy = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16 if training_args.fp16 else torch.float32,
    )


    logger.info(f"Loading reward model from {training_args.reward_model_path}")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, num_labels=1, trust_remote_code=model_config.trust_remote_code
    )


    logger.info("Initializing value model from reward model")
    value_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, num_labels=1, trust_remote_code=model_config.trust_remote_code
    )

    if training_args.rewrite_model_path is None:
        raise ValueError("rewrite_model_path must be specified")
    logger.info(f"Loading rewrite model from {training_args.rewrite_model_path}")
    rewrite_model = AutoModelForCausalLM.from_pretrained(
        training_args.rewrite_model_path,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16 if training_args.fp16 else torch.float32,
    )
    rewrite_model.eval()
    for param in rewrite_model.parameters():
        param.requires_grad = False
    logger.info("Rewrite model frozen")

    if training_args.target_model_path is None:
        raise ValueError("target_model_path must be specified")
    logger.info(f"Loading target model from {training_args.target_model_path}")
    target_model = AutoModelForCausalLM.from_pretrained(
        training_args.target_model_path,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16 if training_args.fp16 else torch.float32,
    )
    target_model.eval()
    for param in target_model.parameters():
        param.requires_grad = False
    logger.info("Target model frozen")

    if training_args.downgrade_model_path is None:
        raise ValueError("downgrade_model_path must be specified")
    logger.info(f"Loading downgrade model from {training_args.downgrade_model_path}")
    downgrade_model = AutoModelForCausalLM.from_pretrained(
        training_args.downgrade_model_path,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16 if training_args.fp16 else torch.float32,
    )
    downgrade_model.eval()
    for param in downgrade_model.parameters():
        param.requires_grad = False
    logger.info("Downgrade model frozen")


    logger.info("Loading HarmBench dataset")
    dataset_path = "/hf_datasets/HarmBench/standard/train-00000-of-00001.parquet"
    df = pd.read_parquet(dataset_path)
    dataset = Dataset.from_pandas(df)
    logger.info(f"Loaded {len(dataset)} examples from HarmBench")


    def tokenize_function(examples):
        outputs = tokenizer(
            examples["Behavior"],
            padding=False,
            truncation=True,
            max_length=256,
        )
        return {"input_ids": outputs["input_ids"]}

    logger.info("Tokenizing dataset")
    train_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=4,
    )
    logger.info(f"Training dataset size: {len(train_dataset)}")

    logger.info("Initializing RedTeamPPOTrainer")
    trainer = RedTeamPPOTrainer(
        config=training_args,
        processing_class=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        rewrite_model=rewrite_model,
        target_model=target_model,
        downgrade_model=downgrade_model,
        train_dataset=train_dataset,
    )

    logger.info("Starting training")
    trainer.train()

    logger.info(f"Saving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)

