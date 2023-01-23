from .libcst_utils import FunctionExtractor
import libcst as cst
import os
import argparse
import json

parser = argparse.ArgumentParser()

parser.add_argument(
    "--folder",
    help="the root folder where to scrape code",
    required=True
)

parser.add_argument(
    "--output",
    help="the output folder where to export the output file",
    required=True
)

def extract_functions(code):
    try:
        tree = cst.parse_module(code)
        fe = FunctionExtractor()
        tree.visit(fe)
        return fe.functions
    except:
        return []

def scrape_folder(rootdir):
    extracted_functions = []
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith('.py'):
                with open(os.path.join(subdir, file)) as pf:
                    try:
                        code = pf.read()
                        extracted_functions.extend(extract_functions(code))
                    except:
                        print('encoding error')

    return extracted_functions


if __name__ == "__main__":
    args = parser.parse_args()
    rootdir = args.folder
    output = args.output
    functions = scrape_folder(rootdir)
    with open(os.path.join(output, "extracted_functions.json"), "w") as out:
        json.dump(functions, out)
