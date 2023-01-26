# CMI-Finder
This artifact is a full functional and reusable implementation of the approach and results presented in the paper "When to Say What: Learning to Find Condition-Message Inconsistencies" https://link.to.paper. The goal of CMI-Finder is to automatically detect condition-message inconsistencies. An example of condition message inconsistency is given below where the operator "or" in the condition is inconsistent with the logic of the message (or --> and):
```Python
if len(bits) != 4 or len(bits) != 6 :
    raise template.TemplateSyntaxError("%r takes exactly\
        four or six arguments (second argument must be ’\
            as’)" % str(bits[0]))
```
The package includes all of the data and code used in the study, as well as utility scripts for visualizing and exporting our findings as they appear in the paper.

## How to use the artifact?
You can import and test the artificat in three different ways:
1. [In our shared docker](#docker-setup)
2. [As a pyhon package through command line](#python-package-setup)

## Docker setup
Before you start this setup, make sure docker is installed on your host machine. If not, please refer to: https://docs.docker.com/get-docker/

### Step 1: Load image
Load the docker image that we share in the folder dockers at the root of repository.
```
docker image load -i ./dockers/cmi.image
```
### Step 2: run and attach 
run and attach the image

### Step 3 : activate the virtual environement
```
cd CMI-Finder
source .venv/bin/activate
```
## Python package setup
In your host machine, navigate to the root of this repository and execute the following:
 ### Step 1: 
 create a new virtual environment using python3.8 or higher. In the following example, we create a virtual environement named .venv
 ```
 python3.8 -m venv .venv
 ```
 ### Step 2: 
 activate the environement (make sure you are in the parent directory of .venv)
 ```
 source .venv/bin/activate
 ```
 ### Step 3: 
 install requirements by using our requirements.txt file located at the root of this repository
 ```
 pip install --upgrade pip
 pip install -r requirements.txt
 ```

 ### step 4:
 install our package cmi-finder
 ```
 pip install .
 ```
## Usage

### **Data collection**
In this step cmi-finder either scrapes randomly a configurable number of repositories or it clones a list of repositories given by the user.

* <b>Option1: Scraping random repositories from GitHub. </b>The following command will scrape 20 repos randomly from github and save them in ./output_folder
    ```
    python -m data_collection.scrape --strategy random --size 20 --output ./demo_repos
    ```
* <b> Option2: Scraping a list of repositories from GitHub. </b> The following command will scrape the list of repositories given in the file target_repos.txt and save them to the folder ./output_folder

    ```
    python -m data_collection.scrape --strategy list --strategy_arg ./demo_repos/target_repos.txt --output ./demo_repos
    ```

<b>Note: </b> all the used folders should exist priorly

### **Data extraction**
In this step cmi-finder extracts functions from all python files given in a directory and all its subtree then extracts condition-message statements from those functions.

* <b> Step1: Extract functions. </b>
The following command extracts all functions from in all python files in the tree of folder ./demo_repos and outputs the results into the folder ./demo_data

    ```
    python -m data_collection.extract_functions --folder ./demo_repos --output ./demo_data
    ```

* <b>Step2: Extract statements. </b>
The following command will extract condition-message statements from the list of functions saved in the file ./demo_data/extracted_functions.json using 16 cpus then saves it to the folder ./demo_data

    ```
    python -m data_collection.extract_data --source ./demo_data/extracted_functions.json -n 16 --output ./demo_data
    ```

### **Data generation**
In this step cmi-finder generates inconsistent condition-message statements from the previously collected likely consistent statements. cmi-finder offers 6 generation techniques. You can invoke all of them at once or each strategy individually. Data generation depend on the existence of a file containig the list of extracted condition message pairs. If you executed the previous steps in data generation, that file is already created. Thus, you can execute what follows.

* <b>Condition mutation. </b>The bellow command executes the condition mutation strategy on the list of condition-message statements given in the file ./demo_data/condition_message_pairs.json using 16 cpus and outputing the results to the folder ./demo_data
    ```
    python -m data_generation.generate --strategy condition --file ./demo_data/condition_message_pairs.json -n 16 --output ./demo_data
    ```
    Similarly the same can be done for the following generation strategies:
* <b>Message mutation </b>
    ```
    python -m data_generation.generate --strategy message --file ./demo_data/condition_message_pairs.json -n 16 --output ./demo_data
    ```

* <b>Random mutation </b>
    ```
    python -m data_generation.generate --strategy random --file ./demo_data/condition_message_pairs.json -n 16 --output ./demo_data
    ```

    Exceptionaly for this strategy, if the generated data is going to be used for training the triplet model, the user should run the following instead of the above:
    ```
    python -m data_generation.generate --strategy random_triplet --file ./demo_data/condition_message_pairs.json -n 16 --output ./demo_data
    ```
* <b>Pattern mutation </b>

    ```
    python -m data_generation.generate --strategy pattern --file ./demo_data/condition_message_pairs.json -n 16 --output ./demo_data
    ```

* <b>Codex mutation </b>

    ```
    python -m data_generation.generate --strategy codex --file ./demo_data/condition_message_pairs.json -n 16 --output ./demo_data
    ```

* <b>Embedding mutation. </b>
 This strategy in particular needs a fasttext model to calculate embeddings. We give a pretrained fasttext model in ./models/emebdding/embed_if_32.mdl

    For this step it is recommended to use one cpu only.

    ```
    python -m data_generation.generate --strategy embed --file ./demo_data/condition_message_pairs.json -n 1 --output ./demo_data --model ./models/embedding/embed_if_32.mdl/embed_if_32.mdl
    ```

* <b>All mutations at once </b>
The following command will apply all mutation on the given data
    ```
    python -m data_generation.generate --strategy all --file ./demo_folder/condition_message_pairs.json -n 1 --output ./demo_folder --model ./models/embedding/embed_if_32.mdl/embed_if_32.mdl
    ```
### **Data preparation**
This step prepares the data collected and generated to be used for training by different neural models.

* <b>Preparing data for BILSTM.</b> The below command prepares the data for the BILSTM model. The command read the data files paths saved in the files ./demo_data/data_paths.json and outputs the results to the folder ./demo_data

    The content of the file ./demo_data/data_paths.json is a dictionary of of the paths of different data files. When creating your own files, make sure to respect the name of the keys as presented in the following example:
    ```Json
    {
        "condition": "test_output_folder/condition_inconsistent_data.json",
        "message": "./demo_data/message_inconsistent_data.json",
        "pattern": "./demo_data/pattern_inconsistent_data.json",
        "embed": "./demo_data/embed_inconsistent_data.json",
        "random": "./demo_data/random_inconsistent_data.json",
        "random_triplet":"./demo_data/random_triplet_inconsistent_data.json",
        "codex": "./demo_data/codex_inconsistent_data.json",
        "consistent": "path/to/some/consistent/data",
        "inconsistent": "path/to/some/inconsistent/data"
    }
    ```
    In the command, we also sepcify the length of sequence of tokens that we want and the vector size depending on the embedding model (default 32) and the embedding model (fasttext)

    ```
    python -m preprocessing.prepare_data --model bilstm --sources ./demo_data/data_paths.json --output ./demo_data --length 64 --vector32
    ```

* <b>Preparing data for Triplet.</b> The below command prepares the data for the triplet model. The command read the data files path saved in the files ./data_paths.json and outputs the results to the folder ./output_folder. In the command, we also sepcify the length of sequence of tokens that we want, the vector size depending on the embedding model (default 32) and the embedding model (fasttext)

    ```
    python -m preprocessing.prepare_data --triplet triplet --sources ./demo_data/data_paths.json --output ./output_folder --length 32 --vector32
    ```

* <b>Preparing data for CodeT5.</b> The below command prepares the data for the CodeT5 model. The command read the data files path saved in the files ./data_paths.json and outputs the results to the folder ./output_folder. 

    ```
    python -m preprocessing.prepare_data --model codet5 --sources ./data_paths.json --output ./output_folder --length 64 --vector32
    ```

### **Train the models**
In this part, we will use cmi-finder to train neural models to detect inconsistent condition-message statements.

* <b>Train BILSTM </b>
    ```
    python -m neural_models.train --model bilstm --class0 test_output_folder/vectorized_consistent.npy --class1 test_output_folder/vectorized_inconsistent.npy --output test_output_folder/
    ```
* <b>Train CodeT5 </b>

    ```
    python -m neural_models.train --model codet5 --class0 test_output_folder/codet5_formatted_data.jsonl --class1 None --output test_output_folder/
    ```
* <b>Train the triplet model</b>
  ```
    python -m
    ```

### **Test the models**
