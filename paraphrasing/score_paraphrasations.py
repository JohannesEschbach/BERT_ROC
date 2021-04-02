import os
import csv
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from uuid import UUID
import pandas as pd
from typing import Iterable, Dict, List
from tqdm import tqdm

from utils import minimum_edit_distance, is_parsable


def paraphrasation_score(original: str, paraphrasation: str) -> int:
    if original == paraphrasation:
        return 0
    ps = PorterStemmer()
    original_vec = [ps.stem(token) for token in word_tokenize(original)]
    paraphrasation_vec = [ps.stem(token)
                          for token in word_tokenize(paraphrasation)]
    edit_distance = minimum_edit_distance(original_vec, paraphrasation_vec)
    sym_diff = len(set(original_vec).difference(paraphrasation_vec)) \
               + len(set(paraphrasation_vec).difference(original_vec))
    return min(edit_distance, sym_diff)


def uuid_dict(rows: Iterable[Iterable[str]]) -> Dict[str, List[str]]:
    return {row[0]: [cell
                     for cell in row[1:]
                     if not is_parsable(cell, int)]
            for row in rows
            if len(row) and is_parsable(row[0], UUID)}


def score_paraphrasations(dataset: str = "cloze_test",
                          wd: str = os.path.realpath(os.path.join(os.getcwd(),
                                                                  "data")),
                          lang_input: str = "en",
                          lang_target: str = "en-GB") -> pd.DataFrame:
    nltk.download('punkt')
    files = [name for name in os.listdir(wd)
             if name.startswith(f"{dataset}.")
             and name.endswith(f"_{lang_target}.csv")]
    lang_pipelines = [filename.split(".")[-2].split("_")
                      for filename in files]
    statistics: pd.DataFrame = None
    original_path = os.path.join(wd, f"{dataset}.csv")
    with open(original_path) as original_file:
        reader = csv.reader(original_file)
        original_data = uuid_dict(reader)
    for languages in (lang_pipelines):
        print("Evaluating paraphrasation score of", tuple(languages[:-1]))
        path = os.path.join(wd, f"{dataset}.{'_'.join(languages)}.csv")
        with open(path) as para_file:
            reader = csv.reader(para_file)
            para_data = uuid_dict(reader)
        scores = []
        for key, para_cells in tqdm(para_data.items()):
            if key not in original_data:
                print(f"{key} not found in {original_path}!")
                continue
            original_cells = original_data[key]
            for original, para in zip(original_cells, para_cells):
                scores.append(paraphrasation_score(original, para))
        this_stats = pd.DataFrame({tuple(languages[:-1]):
                                   scores}).describe().T
        print(this_stats)
        if statistics is None:
            statistics = this_stats
        else:
            statistics = pd.concat([statistics, this_stats])
        print()
    print(statistics)
    return statistics


if __name__ == '__main__':
    score_paraphrasations()
