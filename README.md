
# Brief overview
This repository contains the codes of deepSAGE, a new GLR approach for road type classification of Johannesburg City Road networks.
deepSAGE aims to address the limitations of the state-of-the-art GRL approaches that often performs embeddings on the raw features, which may not be accurate. Therefore, deepSAGE is a multi-stage GRL approach that first obtains the compact feature representation of each road segment (node) using the Deep AutoEncoder (DAE) model. The second stage of deepSAGE introduces RankSAGE, a novel GRL approach that employs an importance based neighbourhood sampling strategy during aggregation. RankSAGE employs a similar GRL layer to GraphSAGE, however, the difference lies in the neighbourhood sampling strategy. Some parts of the code (graph extraction and raw feature generation) were obtained in: https://github.com/zahrag/GAIN

# Packages
Run the following command to install necessary packages needed to run the scripts
* first create a conda environment by running: conda create -deepsage python = 3.7
* Then, activate the newly created environment: conda activate deepsage
* Thereafter, run: pip install -r requirements.txt

# Road network graph extraction
Run the script "01_graph_extraction.py" to extract the road network graph dataset from Open Street Maps.
* Inside the script, define the path where you wish to save the extracted road networks graph dataset. 

# Categorical feature representation
Run the script "02_categorical_feature_extraction.py" to train and select the optimal Deep Neural Networks (DNNE) Entity embedding model.
* Define the base path where the extracted graph dataset is saved (line 11).
* Define the output path where the DNNE model and its artifacts are saved. 
* Running this script will train the DNNE model at various parameters, generate the perfomance report and then select the optimal 
  parameters based on the defined evaluation metrics.

# Stage one: Embedding with Deep AutoEncoder (DAE) model
Run the script "03_deep_auto_encoder_embeddings.py" to train and select the optimal DAE model parameters.
* Define the base path where the graph with DAE features is saved (line 12). 
* Define the output path where the DAE model and its artifacts are saved.
* Running this script will train the DAE model at various parameters, generate the perfomance report and then select the optimal 
  parameters based on the defined evaluation metrics.

# Stage two: Embedding with RankSAGE GRL model
Run the script "04_graph_representation_learning.py" to train various GLR approaches (GCN, GAT, GraphSAGE and RankSAGE)
* Define the base path where the perfomance of each GRL model is saved (line 19).
* Running this script will train a selected GRL model at various parameters, generate the perfomance report and then select the optimal 
  parameters based on the defined evaluation metrics.


