# Paraphrasing

## Data generation

`paraphrase.py` is used to create paraphrasations.
The `fixed_hops` need to be empty for single-pivot pipelines,
but can be seeded with existing one or more pivot languages if existing pipelines shall be used to build upon.
(Change it to your needs.)
Data is read from and written from the CSV files in `data/`.

In order to obtain the translations from DeepL correctly, the following external tools are required:

- the [Tor Browser Bundle](https://www.torproject.org/download/)
- the [Mozilla Geckodriver](https://github.com/mozilla/geckodriver)

as we use [tbselenium](https://github.com/webfp/tor-browser-selenium) to interact with the translation interface.

## Evaluation scores

To compute the evaluation scores, simply run `score_paraphrasations.py`.

This will generate
- `paraphrasing_score.csv` showing a summary on the scores of the different pipeline datasets
- `merged_score.csv`, a similar summary but for the mixed dataset
- `cloze_test.mixed_paraphrased.csv`, the mixed dataset itself and
- `cloze_test.mixed_paraphrased.json`, which records the pipeline and score for each phrase of the mixed dataset.

`scores-without-blacklisting` contains the respective records for the mixed dataset before excluding `ja` and `zh`.

## Additional tools

We used `validate_cleanup.py` on some of the other datasets to bring the CSV files into a common format.
`intersection_detect.py` 
