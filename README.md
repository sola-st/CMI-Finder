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
2. [As a pyhon package through command line](#python-package-cli)
3. [Through jupyter notebooks](#jupyter-notebooks)

## Docker setup

## Python package cli
### I. Installation
 #### Step1: 
 create a new virtual environment using python3.8 or higher. In the following example, we create a virtual environement named .venv
 ```
 python3.8 -m venv .venv
 ```
 #### Step2: 
 activate the environement (make sure you are in the parent directory of .venv)
 ```
 source .venv/bin/activate
 ```
 #### Step3: 
 install requirements by using our requirements.txt file located at the root of this repository
 ```
 pip install --upgrade pip
 pip install -r requirements.txt
 ```

 #### step4
 install our package cmi-finder
 ```
 pip install .
 ```
### II. Usage

#### Data collection
In this step cmi-finder either scrapes randomly a configurable number of repositories or it clones a list of repositories given by the user.

<b> Scraping random repositories from GitHub </b>

The following command will scrape 20 repos randomly from github and save them in ./output_folder
```
python -m data_collection.scrape --strategy random --size 20 --output ./output_folder
```
<b> Scraping a list of repositories from GitHub </b>

The following command will scrape the list of repositories given in the file target_repos.txt and save them to the folder ./output_folder

```
python -m data_collection.scrape --strategy list --strategy_arg target_repos.txt --output .output_folder
```

<b>Note: </b> all the used folders should exist priorly
#### Data extraction
In this step cmi-finder extracts functions from all python files given in a directory and all its subtree then extracts condition-message statements from those functions.

<b> Extract functions </b>
The following command extracts all functions from in all python files in the tree of folder ./scraped_repos and outputs the results into the folder ./data_files

```
python -m data_collection.extract_functions --folder ./scraped_repos --output ./data_files
```

<b>Extract statements </b>
the following command will extract condition-message statements from the list of functions saved in the file ./data_files/extracted_functions using 16 cpus then saves it to the folder ./output_folder

```
python -m data_collection.extract_data --source ./data_files/extracted_functions -n 16 --output ./output_folder
```

#### Data generation
In this step cmi-finder generates inconsistent condition-message statements from the previously collected likely consistent statements. cmi-finder offers 6 generation techniques. You can invoke all of them at once or each strategy individually.

<b>Condition mutation </b>
The bellow command executes the condition mutation strategy on the list of condition-message statements given in the file ./output_folder/condition-message-pairs.json using 16 cpus and outputing the results to the folder ./output_folder
```
python -m data_generation.generate --strategy condition --file ./output_folder/condition-message-pairs.json -n 16 --output ./output_folder
```

<b>Message mutation </b>
The bellow command executes the message mutation strategy on the list of condition-message statements given in the file ./output_folder/condition-message-pairs.json using 16 cpus and outputing the results to the folder ./output_folder
```
python -m data_generation.generate --strategy message --file ./output_folder/condition-message-pairs.json -n 16 --output ./output_folder
```

<b>Random mutation </b>
The bellow command executes the random mutation strategy on the list of condition-message statements given in the file ./output_folder/condition-message-pairs.json using 16 cpus and outputing the results to the folder ./output_folder
```
python -m data_generation.generate --strategy random --file ./output_folder/condition-message-pairs.json -n 16 --output ./output_folder
```

<b>Pattern mutation </b>
The bellow command executes the pattern mutation strategy on the list of condition-message statements given in the file ./output_folder/condition-message-pairs.json using 16 cpus and outputing the results to the folder ./output_folder
```
python -m data_generation.generate --strategy pattern --file ./output_folder/condition-message-pairs.json -n 16 --output ./output_folder
```

<b>Codex mutation </b>
The bellow command executes the codex mutation strategy on the list of condition-message statements given in the file ./output_folder/condition-message-pairs.json using 16 cpus and outputing the results to the folder ./output_folder
```
python -m data_generation.generate --strategy codex --file ./output_folder/condition-message-pairs.json -n 16 --output ./output_folder
```

<b>Embedding mutation </b>
The bellow command executes the embedding mutation strategy on the list of condition-message statements given in the file ./output_folder/condition-message-pairs.json using 1 cpus and outputing the results to the folder ./output_folder. This strategy in particular needs a fasttext model to calculate embeddings. We give a pretrained fasttext model in ./models/emebdding/embed_if_32.mdl

For this step it is recommended to use one cpu only.
```
python -m data_generation.generate --strategy embedding --file ./output_folder/condition-message-pairs.json -n 1 --output ./output_folder --model /models/emebdding/embed_if_32.mdl
```

<b>All mutations at once </b>
The following command will apply all mutation on the given data
```
python -m data_generation.generate --strategy all --file ./output_folder/condition-message-pairs.json -n 1 --output ./output_folder --model /models/emebdding/embed_if_32.mdl
```

## Jupyter notebooks