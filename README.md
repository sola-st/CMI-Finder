# CMI-Finder
This artifact is a full functional and reusable implementation of the approach and results presented in the paper "When to Say What: Learning to Find Condition-Message Inconsistencies" https://link.to.paper. The goal of CMI-Finder is to automatically detect condition-message inconsistencies. An example of condition message inconsistency is given below:
```Python
if condition:
    raise Exception("message")
```
The package includes all of the data and code used in the study, as well as utility scripts for visualizing and exporting our findings as they appear in the paper.

## Requirements

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
 pip install -r requirements.txt
 ```

### II. Fetching and preparing data files
### III. Using cmi-finder to collect data
### IV. Using cmi-finder to generate data
### V. Using cmi-finder to train neural models
### III. Using cmi-finder to run neural models
## Jupyter notebooks