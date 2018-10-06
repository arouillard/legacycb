## INTRODUCTION (preaching to the choir)
• Finding good hyperparameter settings for computational models and algorithms is essential
• Rigorous computational science requires rigorous hyperparameter optimization (or sensitivity analysis in cases where outputs cannot be numerically scored)
• This is relevant for any computational analysis, not only machine learning
• When we are uncertain about how to specify parameters (options, settings, inputs, architectures, designs, etc.), we seek quantitative justification for our choices
	○ If the analysis has a performance score, we search for optimal hyperparameters, and we seek to understand the sensitivity of the performance score to changes in the hyperparameters
	○ If the analysis does not have a performance score, we assess whether our interpretation of the results is sensitive to the hyperparameters
• We routinely do computational experiments/investigations of this kind, but they take much time and effort to implement and run
• Our time and effort should be spent on drug discovery research, not on hyperparameter search
• We need an easy-to-use tool that works with GSK's high performance computing systems to automatically perform distributed hyperparameter search for any computational analysis
• hp_search.py aims to accomplish this
• I hope this code will be widely used and increase the productivity of many people, making it easier to leverage large scale computing to do great science

## BASICS
• User provides a script for a given analysis
	○ Script can be written in Python, R, Perl, or conceivably any language
	○ Script 1) reads hyperparameter settings and other configurations from a text file called input.json, 2) executes a single analysis, and, if the results include a performance score, 3) writes the score to a text file called output.json
• User provides metadata for each hyperparameter to optimize or explore in model_config.json
	○ Variable type: categorical, discrete, or continuous
	○ Domain: set of values or range
	○ Transformation: linear or log (recommended if your range spans orders of magnitude)
• User specifies resources required for a single analysis in job_config.json
	○ Cluster: CMS or GSKTech
	○ Processor_type: CPU or GPU
	○ Num_processors
	○ Memory
	○ Time
• User specifies hyperparameter search settings in search_config.json
	○ Search type: line, grid, random, Bayesian Optimization (bopt), or a combination of bopt, random, and grid search
	○ Time
	○ Max number of hyperparameter combinations to try
	○ More…
• hp_search.py executes the hyperparameter search
	○ Jobs are distributed on the CMS or GSKTech high performance computing cluster
	○ Hyperparameter settings and their corresponding performance scores are written to a text file called hp_search_data.txt

## See Hyperparameter_Search_Documentation for full documentation
