# Content
* **induce_noise.ipynb**: A notebook containing the list of conjunctives as well as temporal and locative adverbials to be induced into a dataset, as well as the function to do so.
* **noise_test_set.csv**: A test set with conjunctives as well as temporal and locative adverbials added to the story, but not the ending.
* **noise_test_set_endingsincluded.csv**: A test set with conjunctives as well as temporal and locative adverbials added to the story and the ending.

# Usage Instructions
* Simply run all cells (if you wish to recreate the noise test sets present in the directory).
* OPTIONAL: If you wish to induce the noise words into another data set than the default 'cloze_test.csv', assign the data set's file name to the 'file' variable in the last cell. Go sure the respective file can be found in the datasets directory.