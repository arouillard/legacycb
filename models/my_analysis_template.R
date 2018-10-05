# -*- coding: utf-8 -*-
# Andrew D. Rouillard
# Computational Biologist
# Target Sciences
# GSK
# andrew.d.rouillard@gsk.com

library(jsonlite)

# load model_config
message('loading model_config...')
model_config_path <- commandArgs(TRUE)
message(paste('model_config_path :', model_config_path))
model_config <- read_json(model_config_path)

# print model_config
message('printing model_config...')
for (field in names(model_config)) {
    message(paste(field, ':', model_config[field]))
}

# get save_folder
message('getting save_folder...')
save_folder <- model_config$save_folder
message(paste('save_folder :', save_folder))

# run analysis
message('running my analysis...')
# commands to run your analysis with the variable settings in model_config

# get objective function(s) value(s)
message('getting objective function(s) value(s)...')
# commands to get objective function value/performance score from analysis
# objective = ...
objective <- 100
message(paste('objective function value(s) :', objective))

# save objective function(s) value(s)
message('saving objective function(s) value(s)...')
output_path <- paste0(save_folder, '/output.json')
message(paste('output_path :', output_path))
write_json(objective, output_path)

# done
message('done my_analysis_template.R')
