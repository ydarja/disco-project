# Enhancing Discourse Relation Classification with Attention Mechanisms on Genre-Diverse Data

**Authors:** Marco Flöß, Darja Jepifanova  
**Course:** Discourse Modelling and Processing 

**Abstract**: ....

**Directories**:
  * **data/** - encompasses the [GUM corpus](https://github.com/amir-zeldes/gum/tree/master/rst/dependencies) rsd files in its suggested data split
  * **descritpive_analyses/** - files concerned with distributions of relations
    * *results*: subdirectory with results
    * *relation_support.ipynb*: gets the support of relations in the data splits
    * *relationg_descriptive_stats.ipynb*: examines discoure related information per relations
  * **genre_clustering** - files concerning the data-driven clustering of GUM version 10 genres
    * *genre_clustering.ipynb*: aggregates discourse and text composition information on the genre level and clusters those
    * *results*: subdirectory with results
  * **literature** - encompasses literature that was reference in our paper
  * **plots** - encompasses resulting plots of the discourse relation classifiers
  * **baseline.py** - discourse relation classifier based on [Zeldes & Liu, 2020](https://github.com/ydarja/disco-project/blob/main/zeldes_liu_2020.pdf)'s architecture
  * **data_manager.py** - data preprocessing steps
  * **requirements.txt** - package versions
  * **t5model.py** - discourse relation classifier based on T5-small


**Overleaf template for writing:** https://www.overleaf.com/7156892639kgxrthxdwhcs#014394  
**Excel table for results:** https://docs.google.com/spreadsheets/d/1NCJh5BHctQUtRghe6AkCyE5OlfGpSEksLl2pZMMp1Cg/edit?usp=sharing
