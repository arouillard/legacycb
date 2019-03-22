# -*- coding: utf-8 -*-
"""
Andrew D. Rouillard
Computational Biologist
Target Sciences
GSK
andrew.d.rouillard@gsk.com
"""


import numpy as np


def get_layer_dimensions(input_dimension, min_dimension, hidden_layers, first_hidden_layer_scaling_factor):
    # returns a list of layer dimensions of the encoder, including the input layer
    if first_hidden_layer_scaling_factor == 'auto':
        all_dimensions = [round(input_dimension*x) for x in [1.0]]
        remaining_hidden_layers = hidden_layers
    elif type(first_hidden_layer_scaling_factor) != str:
        all_dimensions = [round(input_dimension*x) for x in [1.0, first_hidden_layer_scaling_factor]]
        remaining_hidden_layers = hidden_layers - 1
    else:
        raise ValueError('invalid first_hidden_layer_scaling_factor')
    if remaining_hidden_layers > 0:
        layer_scaling_factor = round(100*(all_dimensions[-1]/min_dimension)**(1/remaining_hidden_layers))/100
        all_dimensions += [round(all_dimensions[-1]/(layer_scaling_factor**(x+1))) for x in range(remaining_hidden_layers)]
    if all_dimensions[-1] != min_dimension:
        all_dimensions[-1] = min_dimension
    return all_dimensions

def create_reporting_steps(steps, firstcheckpoint, maxstepspercheckpoint):
    # returns a numpy array of steps when training error and validation error are computed
    if steps == firstcheckpoint:
        return np.array([steps], dtype='int32')
    else:
        ordersofmagnitude = int(np.floor(np.log10(maxstepspercheckpoint)))
        reporting_steps = np.zeros(0, dtype='int32')
        for orderofmagnitude in range(ordersofmagnitude):
            reporting_steps = np.append(reporting_steps, (10**orderofmagnitude)*np.arange(1, 10, 1, dtype='int32'))
        reporting_steps = np.append(reporting_steps,
                                    np.arange(10**ordersofmagnitude, maxstepspercheckpoint, 10**ordersofmagnitude, dtype='int32'))
        reporting_steps = np.append(reporting_steps, np.arange(maxstepspercheckpoint, steps, maxstepspercheckpoint, dtype='int32'))
        reporting_steps = reporting_steps[np.logical_and(reporting_steps > firstcheckpoint, reporting_steps < steps)]
        reporting_steps = np.concatenate(([firstcheckpoint], reporting_steps, [steps]))
        return reporting_steps

def create_layer_training_schedule(hidden_layers, pretraining_epochs, finetuning_epochs, last_layer_epochs, use_finetuning):
    # returns specifications for each training phase
    training_schedule = []
    phase = 0
    for i in range(hidden_layers):
        training_schedule.append({'hidden_layer':i+1, 'finetuning_run':0, 'epochs':pretraining_epochs, 'phase_id':phase})
        phase += 1
        if use_finetuning:
            training_schedule.append({'hidden_layer':i+1, 'finetuning_run':1, 'epochs':finetuning_epochs, 'phase_id':phase})
            phase += 1
    training_schedule[-1]['epochs'] = last_layer_epochs
    training_schedule[-1]['finetuning_run'] = 1
    return training_schedule
