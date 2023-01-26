import argparse
import json

from .codex_generation import generate_inconsistent
from .condition_message_mutation import apply_condition_mutations, mutate_message
from .embedding_based_mutation import replace_identifiers_batch
from .random_matching import random_matching, random_matching_triplet
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

    statements = list(set([tuple(stmt) for stmt in statements]))

    if strategy == "condition":
        print("Generating {0} mutations".format(strategy))
        mutations = apply_condition_mutations(statements)
    elif strategy == "message":
        print("Generating {0} mutations".format(strategy))
        mutations = [((c, m), mutate_message(m)) for c, m in statements]
    elif strategy == "pattern":
        print("Generating {0} mutations".format(strategy))
        mutations = pattern_mutation(statements)
    elif strategy == "embed":
        print("Generating {0} mutations".format(strategy))
        mutations = replace_identifiers_batch(statements[:10000], path_to_model)
    elif strategy == "random":
        print("Generating {0} mutations".format(strategy))
        mutations = random_matching(statements)
    elif strategy == "codex":
        print("Generating {0} mutations".format(strategy))
        api_key = getpass.getpass("Please provide your openai api key(paste it here):")
        mutations = generate_inconsistent(statements, api_key=api_key)
    elif strategy =="random_triplet":
        print("Generating {0} mutations".format(strategy))
        mutations = random_matching_triplet(statements)
    elif strategy == "all":
        print("Generating {0} mutations".format(strategy))
        mutations = {
            "condition": apply_condition_mutations(statements),
            "message": [((c, m), mutate_message(m)) for c, m in statements],
            "pattern": pattern_mutation(statements),
            "embed": replace_identifiers_batch(statements, path_to_model),
            "random": random_matching(statements),
            "codex": generate_inconsistent(statements)
        }
    else:
        raise ValueError("Strategy should be one of condition, message, pattern, embed, random, random_triplet, codex, all")

    print("Saving the results to:", os.path.join(output, strategy+"_inconsistent_data.json"))

    with open(os.path.join(output, strategy+"_inconsistent_data.json"), "w") as exp_data:
        json.dump(mutations, exp_data)


