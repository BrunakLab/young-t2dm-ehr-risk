# A Deep Learning Approach Across Primary and Secondary Care for Early Detection of Young-Onset Type 2 Diabetes
## Introduction
This repository contains the code implementation used in the paper [A Deep Learning Approach Across Primary and Secondary Care for Early Detection of Young-Onset Type 2 Diabetes](INSERT LINK).
We constructed health-event trajectories from different EHR systems spanning both primary and secondary care for detection of young-onset type 2 diabetes in the general population.

## Usage

### Requirements
All code was run within a docker container using the following [Dockerfile](./Dockerfile). We recommend running everything using that file. However, the python version and package versions required can be found in the dockerfile.

### Data structure

Within the folder `data` are example files for all files used in the data pipeline. The files contain no actual data and only exist to show the format and column names used in the pipeline.

* [persons.tsv](./data/persons.tsv) contains information of patient birthdays, sex, status and last event date for that inndividual.
* [diagnoses.tsv](./data/diagnoses.tsv) contains the ICD-10 codes given at the hospital.
* [prescription.tsv](./data/prescription.tsv) contains the prescriptions redeemed at the pharmacies and their ATC code.
* [ydelse.tsv](./data/ydelse.tsv) contains the services performed in primary care.
* [ydelse_mapping.tsv](./data/ydelse_mapping.tsv) contains the mapping between the unmerged service codes and the mapped codes as described in (LINK TO PAPER).


### Generating a duckdb database
The data pipeline can be found in the [Makefile](./scripts/build-db/Makefile) and generates a duckdb database used for querying the data.

### Running models

#### Single run
To run a single experiment use the [main.py](./scripts/main.py). All available arguments can be found in [parsing.py](./diabnet/utils/parsing.py) 

#### Grid search and Results collection
The grid searches used in the publication can be found [here](./configs).

The grid search can be started on an HPC system using torque from the script [Step2-ModelTrainScheduler.py](./scripts/Step2-ModelTrainScheduler.py). Other schedulers have not been implemented, but the script can be modified to accomodate other schedulers.

### Gathering results and bootstrapping performances
[Step3-CollectSearchResults.py](./scripts/Step3-CollectSearchResults.py) can be used to gather the results of a grid search and outputs an overview of the best models.

[Step4-ResultBootstrap.py](./scripts/Step4-ResultBootstrap.py) takes the overview file from `Step3` and bootstraps the performance metrics. 

### Figures from publication
* [performance_plots.R](./notebooks/performance_plots.R) plots the performance figures including figure 3, supplementary figure 4 and 5
* [diabetes_attributions.R](./notebooks/diabetes-attributions.R) plots the attributions plots including figure 4 and 5 and supplementary figure 5 and 6
* [plot_metadata.py](./scripts/metadata/plot_metadata.py) plots the metadata figures including figure 2.
* [region_validation.py](./scripts/metadata/region_assignment.py) assigns regions to individuals and generates supplementary figure 2 and 3

### Issues
Some issues with the data can be debugged using [Step1-ValidateData](./scripts/Step1-ValidateData.py)
