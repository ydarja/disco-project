# Enhancing Discourse Relation Classification with Attention Mechanisms on Genre-Diverse Data

**Authors:** Marco Flöß, Darja Jepifanova  
**Course:** Discourse Modelling and Processing 

**Abstract**: In this work, we investigate discourse relation classification within the context of the Rhetorical Structure Theory (RST) framework, using the newly expanded genre-diverse GUM corpus (version 10.2.0). We introduce a data-driven clustering approach to group genres into five clusters, revealing genre-specific patterns according to the extracted discourse related information. Inspired by previous work on the varying strength of discourse signals, we explore the potential of transformer-based models for discourse relation classification. We compare a bidirectional LSTM baseline (following Zeldes and Liu 2020) with a transformer-based T5-small model with both fine-grained and coarse labels. Our results show that the T5 model consistently outperforms the baseline, particularly in cases where explicit discourse markers are present, and also demonstrates better generalization to less frequent relations. A potential direction for future research is to analyze attention weights in transformer models to assess whether they align with known discourse signals.

**Directories**:
  * **data/** - encompasses the [GUM corpus](https://github.com/amir-zeldes/gum/tree/master/rst/dependencies) rsd files in its suggested data split
  * **descritpive_analyses/** - files concerned with distributions of relations
    * *results/*: subdirectory with results
    * *relation_support.ipynb*: gets the support of relations in the data splits
    * *relationg_descriptive_stats.ipynb*: examines discoure related information per relations
  * **genre_clustering/** - files concerning the data-driven clustering of GUM version 10 genres
    * *genre_clustering.ipynb*: aggregates discourse and text composition information on the genre level and clusters those
    * *results/*: subdirectory with results
  * **literature/** - encompasses literature that was reference in our paper
  * **plots/** - encompasses resulting plots of the discourse relation classifiers
  * **results/** - encompasses a xlsx with model performance results
  * **baseline.py** - discourse relation classifier based on [Zeldes & Liu, 2020](https://github.com/ydarja/disco-project/blob/main/literature/01_Zeldes_Liu2020.pdf)'s architecture
  * **data_manager.py** - data preprocessing steps
  * **requirements.txt** - package versions
  * **t5model.py** - discourse relation classifier based on T5-small

 **Notes on reproducability**:
To reproduce our experiments, please clone this repository to your device and install the required packages from the *requirements.txt*. For the jupyter notebooks of the descriptive analyses and genre clustering one has to adapt the absolute paths to the repository on your machine for the data loading steps. To train on the fine-grained labels, please, change the flag fine_grained to True in the load_data() function. To train on certain clusters, change the flag cluster_group from all to the desired cluster name. Also note, that for the baseline model we don't provide the files with GloVe embeddings, so you should download it yourself and
adjust the code accordingly.