✒ Description
--------------------

This module contains the notebooks related to the trigger words described in the paper. It mostly works with the following three datasets:
* *cloze_test_triggers_only.csv*: Contains only the rows from the original Story Cloze test set where either ending contains trigger words.
* *cloze_test_triggers_removed_only.csv*: Processed *cloze_test_triggers_only.csv* with the trigger words removed from the endings.
* *cloze_test_triggers_synonymized_only.csv*: Processed *cloze_test_triggers_only.csv* with the trigger words synonymized in the endings.
Some additional datasets are:
* *cloze_test_triggers_removed.csv*: Processed *cloze_test.csv* with the trigger words removed from the endings.
* *cloze_test_triggers_synonymized.csv*: Processed *cloze_test.csv* with the trigger words synonymized in the endings.
The last two datasets are mostly ignored, since only a small amount of rows are different from the original test set, whereas the first three datasets are different from each other at each row.


⚙ Prerequisites
--------------------
 
 * !pip install transformers (included in notebooks)

☉ Contents
--------------------

* **📂 dataset_examples**: Contains datasets that were generated during the work on the project.

* **SaliencyMaps.ipynb**: The notebook that displays saliency maps for the trained models. For each model and trigger word a sentence from *cloze_test_triggers_only.csv* and a corresponding sentence from *cloze_test_triggers_removed_only.csv* and from *cloze_test_triggers_synonymized_only.csv* are displayed.

* **TriggersTest.ipynb**: In this notebook, for each testing configuration, models are tested on *cloze_test_triggers_only.csv*, *cloze_test_triggers_removed_only.csv* and *cloze_test_triggers_synonymized_only.csv*.

* **gatherTriggerWords.ipynb**: This is the notebook the execution of which corresponds to steps 1 and 2 of the process to find the trigger words described in chapter 6.3. Trigger Words of the paper.

* **negated_synonymized.csv**: This dataset is a byproduct of the pipeline used to create the hardest test set.

* **saliencies.html**: The output of *SaliencyMaps.ipynb* was manually gathered here.

* **saliencies.pdf**: PDF version of the previous file.

* **triggersTruncateSynonymize.ipynb**: In this notebook, the original Story Cloze test set is processed creating the datasets *cloze_test_triggers_only.csv*, *cloze_test_triggers_removed_only.csv* and *cloze_test_triggers_synonymized_only.csv* as well as *cloze_test_triggers_removed.csv* and *cloze_test_triggers_synonymized.csv* if specified.

* **triggerstruncatesynonymize.ipynb**: This script was used for the pipeline used to create the hardest test set.

⚛ Usage
--------------------

Preferred order of execution of the notebooks is:
1. gatherTriggerWords.ipynb (optional)
2. triggersTruncateSynonymize.ipynb
3. SaliencyMaps.ipynb or TriggersTest.ipynb in no particular order

