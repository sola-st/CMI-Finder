import argparse
import json

from .codex_generation import generate_inconsistent
from .condition_message_mutation import apply_condition_mutations, mutate_message
from .embedding_based_mutation import replace_identifiers_batch
from .random_matching import random_matching
from .pattern_based_mutation import pattern_mutation
from data_collection.utils import run_merge_responses
import os
import getpass

parser = argparse.ArgumentParser()

parser.add_argument(
    "--strategy",
    help="What generation strategy to generate inconsistent statements:[condition, message, pattern, embed, codex, random, all]",
    required=True
)

parser.add_argument(
    "--file",
    help="Path to a file containing the list of pairs of condition message statements in json format",
    required=True
)

parser.add_argument(
    "-n",
    help="number of cpus",
    default= 1 
)

parser.add_argument(
    "--output",
    help="path to the output dir",
    required=True
)

parser.add_argument(
    "--model",
    help="path to the embeding model"
)

if __name__ == "__main__":
    args = parser.parse_args()

    strategy = args.strategy
    file_path = args.file
    n_cpus = args.n
    output = args.output
    path_to_model = args.model

    with open(file_path) as fp:
        statements = json.load(fp)

    if strategy == "condition":
        mutations = apply_condition_mutations(statements)
    elif strategy == "message":
        mutations = [(c, mutate_message(m)) for c, m in statements]
    elif strategy == "pattern":
        mutations = pattern_mutation(statements)
    elif strategy == "embed":
        mutations = replace_identifiers_batch(statements, path_to_model)
    elif strategy == "random":
        mutations = random_matching(statements)
    elif strategy == "codex":
        api_key = getpass.getpass("Please provide your openai api key(paste it here):")
        mutations = generate_inconsistent(statements, api_key=api_key)
    elif strategy == "all":
        mutations = {
            "condition": apply_condition_mutations(statements),
            "message": [(c, mutate_message(m)) for c, m in statements],
            "pattern": pattern_mutation(statements),
            "embed": replace_identifiers_batch(statements, path_to_model),
            "random": random_matching(statements),
            "codex": generate_inconsistent(statements)
        }
    with open(os.path.join(output, strategy+"_inconsistent_data.json"), "w") as exp_data:
        json.dump(mutations, exp_data)


