# Master project
This project was made as part of my master's degree at NTNU. It was used to detect partial contract cheating. The datasets used were PAN20, PAN21, and PAN23.

# Short readme
- pipe.py is the main application and is run with python3 pipe.py (no arguments). Requirements must be installed (start with requirements.txt, but there are some extra for spacy). 
- Experiments define the experiments to be run, including where to find the datasets. 

The current project needs the following folder structure: 
```
datasets
preprocessed_datasets
master/pipe.py
results
```
# Todo
- Do something with the tokenization process. This is a very heavy process. Maybe do all tokenization before everything else? 
