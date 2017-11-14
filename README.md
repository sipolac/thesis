# thesis

This contains the code used for my master's thesis, [Summarizing electricity usage with a neural network](https://drive.google.com/file/d/0B2eLkoWKdSD0enR4N2d2TEdlT0E/view?usp=sharing), which was supervised by [Nigel Goddard](http://homepages.inf.ed.ac.uk/ngoddard/). The thesis was written for the MSc in Artificial Intelligence program at the University of Edinburgh and was submitted in August 2017. It represents roughly 10 weeks of work. For this thesis I was awarded the Informatics Dissertation Prize for Top Performance.

The abstract for the thesis is:

> This project explores whether a neural network is capable of predicting summary statistics of electricity usage for five common household appliances given only the aggregate signal of a smart meter. These appliance-level statistics are needed for many kinds of feedback and analytics provided to energy consumers so they can reduce electricity consumption and save on energy bills. An example of such a statistic is the total energy used by a washing machine in a day. Currently the state-of-the-art approach is to use non-intrusive load monitoring (NILM) to predict appliance-level signals timepoint-by-timepoint, and then compute the statistics using these predictions. However, this is indirect, computationally expensive and generally a harder learning problem.

> The statistics can also be used as input into one of the most successful NILM models, the additive factorial hidden Markov model (AFHMM) with latent Bayesian melding (LBM). This model uses these appliance-level statistics as priors to significantly improve timepoint-by-timepoint predictions. However, the model is currently limited to using national averages, since so far there have been no methods for inferring day- and house-specific statistics. Improved statistics can therefore lead to more effective NILM models.

> Since this type of learning problem is unexplored, we use a dynamic architecture generation process to find networks that perform well. We also implement a new process for generating more realistic synthetic data that preserves some cross-appliance usage information. Results show that a neural network is in fact capable of predicting appliance-level summary statistics. More importantly, most models generalize successfully to houses that were not used in training and validation, with the best-performing models having an error that is less than half the baseline.


To re-run the project:

1. Place the [cleaned CSV data from the REFIT site](https://pure.strath.ac.uk/portal/en/datasets/refit-electrical-load-measurements-cleaned(9ab14b0e-19ac-4279-938f-27f643078cec).html) into a directory data/CLEAN_REFIT_081116/.

2. Run **py/create_data.py**, which creates the data used for the neural network models. This data is output into a directory run/YYYY-MM-DD that is created in the project directory, which then needs to be moved manually to the directory data/for_models/. (This manual process avoids accidentally overwriting.)

3. Run **py/build_models.py**, which builds the models once the modeling data is in the right location. This takes *days* to run on a CPU, but it can by stopped by a keyboard interrupt and restarted at any point with no issues.

A few notes:

- The Jupyter notebooks (\*.ipynb) are used to create graphs, tables and calculations for the paper. It is ad-hoc and very disorganized (sorry!), but the file names are more or less associated with the Python file used to create the data explored in the notebook.

- If the cleaned REFIT data was updated since this code was written, you may need to make modifications to the directory name and to the function ``save_refit_data`` in py/create_data.py.

- The file data/appliances.csv was created "by hand" using the README from the REFIT site.
