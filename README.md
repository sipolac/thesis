# thesis

This contains the code for my thesis for the MSc Artificial Intelligence program at the University of Edinburgh. The thesis is titled "Summarizing electricity usage with a neural network" and is supervised by [Nigel Goddard](http://homepages.inf.ed.ac.uk/ngoddard/).

The abstract is:

> This project explores whether a neural network is capable of predicting summary statistics of electricity usage for five common household appliances given only the aggregate signal of a smart meter. These appliance-level statistics are needed for many kinds of feedback and analytics provided to energy consumers so they can reduce electricity consumption and save on energy bills. An example of such a statistic is the total energy used by a washing machine in a day. Currently the state-of-the-art approach is to use non-intrusive load monitoring (NILM) to predict appliance-level signals timepoint-by-timepoint, and then compute the statistics using these predictions. However, this is indirect, computationally expensive and generally a harder learning problem.

> The statistics can also be used as input into one of the most successful NILM models, the additive factorial hidden Markov model (AFHMM) with latent Bayesian melding (LBM). This model uses these appliance-level statistics as priors to significantly improve timepoint-by-timepoint predictions. However, the model is currently limited to using national averages, since so far there have been no methods for inferring day- and house-specific statistics. Improved statistics can therefore lead to more effective NILM models.

> Since this type of learning problem is unexplored, we use a dynamic architecture generation process to find networks that perform well. We also implement a new process for generating more realistic synthetic data that preserves some cross-appliance usage information. Results show that a neural network is in fact capable of predicting appliance-level summary statistics. More importantly, most models generalize successfully to houses that were not used in training and validation, with the best-performing models having an error that is less than half the baseline.



Before running anything, the cleaned CSV data from the REFIT site, https://pure.strath.ac.uk/portal/en/datasets/refit-electrical-load-measurements-cleaned(9ab14b0e-19ac-4279-938f-27f643078cec).html, needs to be placed in a directory data/CLEAN_REFIT_081116/. Note that the file data/appliances.csv was created "by hand" using the README from the REFIT site.

py/create_data.py creates the modeling data. This data is output into a "run" directory that is created in the project directory. This data then needs to be manually moved to a directory data/for_models/. (This manual process avoids accidentally overwriting.)

py/build_models.py builds the models once the modeling data is in the right location. This takes a *very* long time, but it can by stopped and restarted at any point with no issues.

The code in ipynb/ is used to create graphs and tables for the paper. It is also used to make calculations used in the paper. It is not very organized, but the file name is more or less associated with the Python file used to create the data described in the notebook.

py/utils.py just contains utility functions used by the other Python files.
