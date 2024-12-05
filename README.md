# HPC Tandem Predictions: A Precursor to `sbatch_pred`

## Introduction

This tool is designed to provide accurate predictions for HPC job runtimes and queue times. It uses Machine Learning (ML) models to deliver these predictions with associated uncertainty estimates. This document presents a detailed technical analysis of the models and methodologies used. This codebase is meant as a precursor to a user facing tool, `sbatch_pred`, which provides HPC users with runtime and queue time predictions for their job.

## Overview

This codebase accompanies a paper accepted in the proceedings of [PEARC24](https://pearc.acm.org/pearc24/). In this work, we compared 12 variations of system state features for queue time prediction models. These variations were a result of 3 diferent options for getting a job runtime estimate (a user estimate, the runtime predicted by a separate machine learning model, and the perfect knowledge of the runtime available only after the job has finished), and 4 different options for the way the cluster is understood i.e. *knowledge level* (as a single unified cluster, as a set of isolated partitions, as a set of node-groups available to different queues, or as an amalgamation of partitions and node-groups).

### Model Overview

`sbach_pred` employs two types of models:

1. **Classification Models**: These models categorize jobs into different wait time classes. Historical confusion matrices are maintained to understand the distribution of actual wait times for each predicted class.

2. **Regression Models**: These models predict continuous wait times. The tool introduces random noise into job features and distributions of actual wait times from similar jobs to quantify uncertainty.

### Results Analysis

Our analysis shows that:

- The accuracy of queue time predictions improves significantly when the model is given knowledge of the system state (e.g. number of jobs waiting in a queue, number of nodes currently in use, etc.),  with similar improvements for classification and regression models.
- Using perfect knowledge of job runtime improves performance when compared to models trained with the user estimate of job runtime.
- Using the job runtime predicted with ML decreased the error in queue time predicted by the regression models
- Using a combination of partition and node-level knowledge resulted in the least error in regression models and the highest accuracy in classification models

## Installation

To install the package, follow these seps:

1. **Clone the repository**

2. **Set up the conda environment and install `sbatch_pred`**

- Create the conda environment using the provided `environment.yml` file:

  `conda env create -f environment.yml`
  
  *This will set up the conda environment and install the `sbatch_pred` Python package contained in this repository.*
  
- Activate the conda environment:
  
  `conda activate sbatch_pred`
  
## Getting Started with Notebooks

*Note: Though this repository is set up to allow you to get runtime predictions for HPC jobs, a thorough approach to this task is already available at [https://github.com/NREL/eagle-jobs](https://github.com/NREL/eagle-jobs). As such, the remainder of this guide is focused on Queue Time prediction and Uncertainty Analysis*

Open Jupyter Lab:

`jupyter lab`

### Queue Time Prediction Notebooks
- `system_state`: Calculates system state features to use in the queue time prediction models. System state features are calculated in two groups:
  - *Queue State Features*: Description of the state of the queue based on characteristics of jobs that have been submitted but have not yet started.
  - *System Utilization Features*: Description of the state of the system based on jobs that have started and are currently running.
  
  This notebook requires four separate datasets to run: `slurm_data.parquet`, `nodes.parquet`, `partitions.parquet`, and `predicted_runtime.parquet`. These datasets are already provided in the `data` directory. Running this notebook will take a few hours and will generate the data files needed in the `model_training` notebook. However, it is not necessary to complete this step, as **this data is already available** in the `data/model_data` directory.
- `model_training`: Uses XGBoost to train either regression or classification models. The desired knowledge level (`cluster`, `partition`, or `node`) and type of wallclock knowledge (`user estimate`, `runtime with ML`, or `perfect knowledge`). Results will be saved in the `data/results` directory. (*Note: Sample results are already available*). This notebook can also be used to optimize model hyperparameters via Optuna.
- `analyze_results_regression`: Use this notebook to analyze the results of the regression models.
- `analyze_results_classification`: Use this notebook to analyze the results of the classification models.
- `uncertainty_analysis`: This notebook is useful to determine the uncertainty of the regression models in queue time predictions. This work is still under development. Please see the documentation in the notebook for a thorough description of the approach.
  

## Model Overview

`sbach_pred` employs two types of models:

1. **Classification Models**: These models categorize jobs into different wait time classes. Historical confusion matrices are maintained to understand the distribution of actual wait times for each predicted class.

2. **Regression Models**: These models predict continuous wait times. The tool introduces random noise into job features and distributions of actual wait times from similar jobs to quantify uncertainty.

## Detailed Results
### Per-Partition Results for Regression Models
![reg_results](https://github.com/NREL/hpc_tandem_predictions/assets/77375297/c1bee703-10bd-455c-85ca-933ed9d6b9ba)

### Per-Partition Results for Classification Models
![class_results](https://github.com/NREL/hpc_tandem_predictions/assets/77375297/c077bee5-0f62-4ef1-9ff5-c705d4a5908c)

### Confusion Matrices for System-State Feature Set Variations
![cm](https://github.com/NREL/hpc_tandem_predictions/assets/77375297/bc554b4e-52f3-48df-b35c-7c7c3ce8360f)

