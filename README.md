# Acquiring Story Understanding through Training with Short Stories
## Probing Common-Sense Reasoning of fine-tuned BERT Language Models
### Semantics project WS20/21

✒ Description
--------------------
This repository comprises our finished project done over the course of 4 weeks for the Semantics course at the university of Heidelberg in the winter semester of 2020/2021. The goal was to independently and practically apply the aquired knowledge of model training and testing led by a semantically relevant question for machine learning. The findings are recorded in a paper, which is to be presented. 

* maintainers: students of **Computational Linguistics**: Johannes Eschbach, Kamal Eyuboy, Leon Patzig, Raziye Sari
* contact: *<first_name>.<last_name>*@stud.uni-heidelberg.de  


⚙ Prerequisites
--------------------
 
 * !pip install transformers

☉ Contents
--------------------

* **📂 train_and_test**: Contains CleanNotebook.ipynb which loads all needed data & models, trains and tests accordingly

* **📂 datasets**: Contains all needed default datasets of ROC Stories and Cloze training and testing. Automatically or manually created test sets: **negated**, **noise** and **super hard** also included

* **📂 noise**: modifies test set by inducing discourse relation markers as noise in each sentence. Extra test set with endings also induced. 

* **📂 paraphrasing**: modifies test set by paraphrasing each sentence using deepL, a language translation tool.

* **📂 triggers**: creates separate test sets with identified trigger words removed and synonymized

* **📂 super_hard**: gathers all 4 experiments in one single final test set. 

* **📂 salience_mapping**, **📂 weight_changes**: generating of metrices for analysis of findings

* **Models.txt**: Contains Google-drive links to the models     

* **Appendix.pdf**: Collection of all in the paper referenced weight changes, test results and saliency maps  
