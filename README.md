# Transformer-Based Neural Marked Spatio Temporal Point Process Model for Football Match Events Analysis
The NMSTPP model architecture:
![alt text](https://anonymous.4open.science/r/NMSTPP/NMSTPP.png?raw=true)

The code and dataset for the proposed model are contained in the following files.

1. dataset_create.py creates the csv dataset from the json file retrived from the [WyScout Open Access Dataset](https://figshare.com/ndownloader/files/14464685/events.zip) referencing the code for the [Seq2event](https://github.com/statsonthecloud/Soccer-SEQ2Event/blob/main/Seq2Event_Notebook01_DataImport.ipynb) model and the corresponding heatmaps for the grouped actions (pass, dribble, cross, shot, and possession end).

2. train_valid_test_split.py creates the other continouses features and split the csv dataset created in step 1 into to train/valid/test according to the 0.8/0.1/0.1 ratio.

3. NMSTPP.py is the code for training and validating the the NMSTPP model referencing the code for the [Seq2event](https://github.com/statsonthecloud/Soccer-SEQ2Event/blob/main/Seq2Event_Notebook01_DataImport.ipynb) model, and the testing result is also included.

4. forecast.py is the code for forecasting the entire dataset.

# Introduction
With recently available football match event data that record the detail of the football match, analysts and researchers have a great opportunity to develop new performance metrics, gain insight, and evaluate key performance.
However, most sports sequential events modeling methods and performance metrics approaches could be insufficient for dealing with such large-scale sequential data, necessitating a more complex model and holistic performance metric. 

To this end, we propose the Transformer-Based Neural Marked Spatio Temporal Point Process (NMSTPP) model for football event data based on the neural temporal point processes (NTPP) framework. In the experiments, our model outperformed the prediction performance of the baseline models. Furthermore, we have proposed the holistic possession utilization score (HPUS) metric for a more comprehensive football possession analysis. For verification, we examined the relationship with football teams' final ranking, average goal score, and average xG over a season, the average HPUS showed significant correlations regardless of not using details shot or goal information.

# References
Anonymous. "Transformer-Based Neural Marked Spatio Temporal Point Process Model for Football Match Events Analysis". 
