import argparse
import os
from .extract_statements import extract_batch_, filter_batch, remove_extra_para, remove_extra_space, exclude_empty_strings
from .utils import run_merge_responses
import json

parser = argparse.ArgumentParser()
parser.add_argument(
    "--source", 
    help="specify the the path to the source file or folder to file(s) containing a list of extracted fucntions",
    required = True
    )

parser.add_argument(
    "-n",
    help="number of cpus to use for extraction",
    default=1
)

parser.add_argument(
    "--output",
    help = "directory where to save the extracted data file",
    required=True
)

def extract_condition_message_pairs(source, output_dir, cat="file", n=1):
    functions = []
    print("Loading functions list")
    if cat == "file":
        with open(source) as sf:
            functions = json.load(sf)
    elif cat == "dir":
        functions_files = [d for d in os.listdir(source) if d.endswith(".json")]
        for func_file in functions_files:
            with open(os.path.join(source, func_file)) as ffd:
                functions.extend(json.load(ffd))
    
    print("Extracting condition-message statements")
    pairs = run_merge_responses(functions, extract_batch_,n_cpus_a=n)

    print("Cleaning and simplifying codition message statements")
    pairs_inline = []
    for p in pairs:
        stmts = p[0]
        func_def = p[1]
        for stmt in stmts:
            pairs_inline.append((stmt, func_def))

    pairs_inline = [pi[0] for pi in pairs_inline]
    simple_pairs = []

    for c, m in pairs_inline:
        if type(c[0]) == str:
            condition = c[0]
        else:
            condition = '(' + c[0][0] + ' ' + c[0][1] + ')' 
        for ci in c[1:]:
            if type(ci) == str:
                condition += ' and ' + '(' + ci + ')'
            else:
                condition += ' and ' + '(' + ci[0] + ' ' + ci[1] + ')'
        
        simple_pairs.append((condition, m))

    data_pairs_1 = [(remove_extra_para(c), m) for c,m in simple_pairs]
    data_pairs_2 = [(remove_extra_space(c), m) for c, m in data_pairs_1]
    print_raise_pairs = []
    for c, m in data_pairs_2:
        if '(' in m:
            if 'print' in m[: m.find('(')] or 'log' in m[: m.find('(')] or 'raise' in m[: m.find('(')]:
                print_raise_pairs.append((c, m))
    not_empty_strings = exclude_empty_strings(print_raise_pairs)
    
    print("{0} pairs were extracted, writing data to: ".format(len(not_empty_strings)), os.path.join(output_dir, "extracted_condition_message_pairs.json"))
    with open(os.path.join(output_dir, "extracted_condition_message_pairs.json"), 'w') as edj:
        json.dump(not_empty_strings, edj)

if __name__ == "__main__":
    args = parser.parse_args()
    source = args.source
    output = args.output
    n = int(args.n)
    if os.path.isfile(source):
        extract_condition_message_pairs(source, output, cat="file", n=n)
    elif os.path.isdir(source):
        extract_condition_message_pairs(source, output, cat="dir", n=n)
    else:
        raise FileNotFoundError("Could not find the specified path")