## INTRODUCTION
### Finding good hyperparameter settings for computational models and algorithms is essential
### Rigorous computational science requires rigorous hyperparameter optimization (or sensitivity analysis in cases where outputs cannot be numerically scored)
### This is relevant for any computational analysis, not only machine learning
### When we are uncertain about how to specify parameters (options, settings, inputs, architectures, designs, etc.), we seek quantitative justification for our choices
#### If the analysis has a performance score, we search for optimal hyperparameters, and we seek to understand the sensitivity of the performance score to changes in the hyperparameters
#### If the analysis does not have a performance score, we assess whether our interpretation of the results is sensitive to the hyperparameters
### We routinely do computational experiments/investigations of this kind, but they take much time and effort to implement and run
### Our time and effort should be spent on drug discovery research, not on hyperparameter search
### We need an easy-to-use tool that works with GSK's high performance computing systems to automatically perform distributed hyperparameter search for any computational analysis
### hp_search.py aims to accomplish this
### I hope this code will be widely used and increase the productivity of many people, making it easier to leverage large scale computing to do great science

## BASICS
### User provides a script for a given analysis
### User provides metadata for each hyperparameter to optimize or explore in model_config.json
### User specifies resources required for a single analysis in job_config.json
### User specifies hyperparameter search settings in search_config.json
### hp_search.py executes the hyperparameter search
#### Jobs are distributed on the CMS or GSKTech high performance computing cluster
#### Hyperparameter settings and their corresponding performance scores are written to a text file called hp_search_data.txt

## See Hyperparameter_Search_Documentation for full documentation
