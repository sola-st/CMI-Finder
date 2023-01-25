import torch
from datasets import load_dataset
from transformers import RobertaTokenizer, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from transformers import RobertaTokenizer, RobertaForMaskedLM
from transformers.models.t5 import T5ForConditionalGeneration
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from transformers.utils import logging
import signal
import random
import numpy as np
from prettytable import PrettyTable
import json
import os

class FineTuneModel():

    """
    A class wrapping the necessary code to instantiate codet5 from transformers and finetune it or retrain it

    Attributes:
    initial_model: str
        the name of the model that you want to finetune

    training_config: str
        Path of the training configuration file

    output_dir: str
        Path to save the model to

    data_path: str
        directory of training and validation data

    data_cache_path: str
        path to cache preprocessed data for later use

    workers: str
        number of workers used to preprocess and tokenize the data

    seed: str
        the seed used to randomize the initialization of the parameters


    """

    def __init__(self, initial_model, training_config, output_dir, data_path, data_cache_path, workers, seed):
        self.initial_model = initial_model
        self.training_config = training_config
        self.output_dir = output_dir
        self.data_path = data_path
        self.data_cache_path = data_cache_path
        self.workers = workers
        self.seed = seed
        self.overwrite_cache = True

        if self.training_config is None:
            try:
                self.set_default_config()
                self.__set_training_arguments()
            except Exception as e:
                print(e)
                print("You didn't provide a config file for training and we couldn't use the default one")
                print("create a config file and pass it as a parameter")
                print("you can copy-paste the following configuration")
                print('''{
                    "overwrite_output_dir": true,
                    "do_train": true,
                    "do_eval": true,
                    "per_device_train_batch_size": 4,
                    "per_device_eval_batch_size": 4,
                    "gradient_accumulation_steps": 20,
                    "evaluation_strategy": "steps",
                    "eval_steps": 20,
                    "learning_rate": 5e-05,
                    "weight_decay": 0.001,
                    "max_grad_norm": 1.0,
                    "max_steps": 40,
                    "lr_scheduler_type": "linear",
                    "warmup_steps": 100,
                    "logging_strategy": "steps",
                    "logging_steps": 5,
                    "save_strategy": "steps",
                    "save_steps": 50,
                    "save_total_limit": 1,
                    "save_on_each_node": false,
                    "no_cuda": false,
                    "seed": 42,
                    "fp16": false
                }''')
                
    def set_default_config(self):
        default_config = '''{
            "overwrite_output_dir": true,
            "do_train": true,
            "do_eval": true,
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            "gradient_accumulation_steps": 20,
            "evaluation_strategy": "steps",
            "eval_steps": 20,
            "learning_rate": 5e-05,
            "weight_decay": 0.001,
            "max_grad_norm": 1.0,
            "max_steps": 40,
            "lr_scheduler_type": "linear",
            "warmup_steps": 100,
            "logging_strategy": "steps",
            "logging_steps": 5,
            "save_strategy": "steps",
            "save_steps": 50,
            "save_total_limit": 1,
            "save_on_each_node": false,
            "no_cuda": false,
            "seed": 42,
            "fp16": false
        }'''

        with open("default_config_file.json", "w") as dcf:
            dcf.write(default_config)

        self.training_config = "default_config_file.json"

    def set_train_data_file(self, file_name):
        self.train_file = file_name

    def set_validation_data_file(self, file_name):
        self.validation_file = file_name

    def load_dataset(self):
        assert(hasattr(self, "train_file"))
        assert(hasattr(self, "validation_file"))
        data_files = {
            "train": self.train_file,
            "validation": self.validation_file
        }
        import os
        print(os.getcwd())
        raw_datasets = load_dataset(
            path = "./",
            data_dir = self.data_path,
            data_files = data_files,
            cache_dir = self.data_cache_path
        )

        return raw_datasets

    def __set_training_arguments(self):
        training_args = json.load(open(self.training_config))
        training_args["output_dir"] = self.output_dir
        self.training_args = TrainingArguments(**training_args)
        self.training_args.dataloader_num_workers = self.workers


    def prepare_features(self, examples, input_label="source", target_label="target", max_length=512):
        
        inputs = examples[input_label]
        targets = examples[target_label]
        self.tokenizer.model_max_length = 512
        model_inputs = self.tokenizer(inputs, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True)
        labels = self.tokenizer(targets, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True)

        labels["input_ids"] = [
            [(_l if _l != self.tokenizer.pad_token_id else -100) for _l in label] for label in labels["input_ids"]
        ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def preprocess_training_data(self):
        self.data = self.load_dataset()
        train_dataset = self.data["train"]
        with self.training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                self.prepare_features,
                batched=True,
                num_proc=self.workers,
                remove_columns = train_dataset.column_names,
                load_from_cache_file=not self.overwrite_cache,
                desc="Running tokenizer on train dataset",
        )

        return train_dataset

    def preprocess_validation_data(self):
        self.data = self.load_dataset()
        eval_dataset = self.data["validation"]
        with self.training_args.main_process_first(desc="train dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                self.prepare_features,
                batched=True,
                num_proc=self.workers,
                remove_columns = eval_dataset.column_names,
                load_from_cache_file=not self.overwrite_cache,
                desc="Running tokenizer on train dataset",
        )

        return eval_dataset

class FineTuneCodeT5(FineTuneModel):

    def __init__(self, training_config, output_dir, data_path, data_cache_path, workers, seed):
        super().__init__("codet5-small", training_config, output_dir, data_path, data_cache_path, workers, seed)
        self.model = None
        self.tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')
        self.model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-small')

    def set_seeds(seed):
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed(seed)

    def num_parameters(self):
        model_parameters = self.model.parameters()
        return sum([np.prod(p.size()) for p in model_parameters])

    def count_parameters(self):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad: continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params+=params
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params

    def __trainer(self):
        trainer = Trainer(
        model=self.model,
        args=self.training_args,
        train_dataset=self.preprocess_training_data(),
        eval_dataset=self.preprocess_validation_data(),
        data_collator=DataCollatorForSeq2Seq(
            model=self.model,
            tokenizer=self.tokenizer,
            padding='max_length',
            ),
        tokenizer=self.tokenizer,
        )

        return trainer

    def train_from_scratch(self):
        trainer = self.__trainer()
        trainer.train()

    def train_from_check_point(self):
        trainer = self.__trainer()
        last_checkpoint = get_last_checkpoint(self.output_dir)
        trainer.train(resume_from_checkpoint=last_checkpoint)

    def save_model(self, folder):
        self.model.save_pretrained(os.path.join(folder, "t5_classification_final.mdl"))