import os
import csv
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from uuid import UUID
import pandas as pd
from typing import Iterable, Dict, List, Tuple
from tqdm import tqdm

from utils import minimum_edit_distance, is_parsable


STRINGS_PER_STORY = 6


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
    # print(original, paraphrasation)
    return min(edit_distance, sym_diff)


def write_maxima(maxima: Dict[str, List[Tuple[str, str, float]]],
                 original_data: Dict[str, List[str]],
                 header: List[str]):
    meta: Dict[str, List[str]] = {}
    with open("cloze_test.mixed_paraphrased.csv", "w") as outfile:
        csv_writer = csv.writer(outfile)
        csv_writer.writerow(header)
        for key, entries in maxima.items():
            paras, languages, scores = zip(*entries)
            scores = [int(score) for score in scores]
            assert len(paras) == STRINGS_PER_STORY
            meta[key] = list(zip(list(languages), scores))
            csv_writer.writerow([key] + list(paras) + [original_data[key][-1]])
    with open("cloze_test.mixed_paraphrased.json", "w") as outfile:
        json.dump(meta, outfile)


def uuid_dict(rows: Iterable[Iterable[str]]) -> Dict[str, List[str]]:
    return next(rows), {row[0]: [cell
                                 for cell in row[1:]]
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
    lang_blacklist = ["zh-ZH", "ja-JA"]
    lang_pipelines = [pipeline for pipeline in lang_pipelines
                      if not any(lang in pipeline for lang in lang_blacklist)]
    statistics: pd.DataFrame = None
    original_path = os.path.join(wd, f"{dataset}.csv")
    maxima: Dict[str, List[Tuple[str, str, float]]] = {}
    with open(original_path) as original_file:
        reader = csv.reader(original_file)
        header, original_data = uuid_dict(reader)
    for languages in (lang_pipelines):
        print("Evaluating paraphrasation score of", tuple(languages[:-1]))
        path = os.path.join(wd, f"{dataset}.{'_'.join(languages)}.csv")
        with open(path) as para_file:
            reader = csv.reader(para_file)
            _, para_data = uuid_dict(reader)
        scores = []
        for key, para_cells in tqdm(para_data.items()):
            if key not in original_data:
                print(f"{key} not found in {original_path}!")
                continue
            original_cells = original_data[key]
            if len(original_cells) != len(para_cells):
                print(f"found original_cells of length {len(original_cells)}",
                      original_cells, key)
                print(f"found para_cells of length {len(para_cells)}",
                      para_cells, key)
            if key not in maxima:
                maxima[key] = []
            for i, para in enumerate(para_cells):
                if i >= STRINGS_PER_STORY:
                    continue
                score = paraphrasation_score(original_cells[i], para)
                scores.append(score)
                if len(maxima[key]) <= i:
                    maxima[key].append((para, tuple(languages[:-1]), score))
                elif maxima[key][i][2] < score:
                    maxima[key][i] = (para, tuple(languages[:-1]), score)
            assert len(maxima[key]) == STRINGS_PER_STORY
            assert len(para_cells) == STRINGS_PER_STORY + 1
            assert len(original_cells) == STRINGS_PER_STORY + 1
            assert original_cells[-1] == para_cells[-1]
        this_stats = pd.DataFrame({tuple(languages[:-1]):
                                   scores}).describe().T
        print(this_stats)
        if statistics is None:
            statistics = this_stats
        else:
            statistics = pd.concat([statistics, this_stats])
        print()
    statistics = statistics.sort_values("mean")
    print(statistics)
    statistics.to_csv("paraphrasing_score.csv")
    merged = pd.DataFrame({
        "languages": [languages for values in maxima.values()
                      for para, languages, score in values],
        "score": [score for values in maxima.values()
                  for para, languages, score in values]
    }).describe()
    merged.to_csv("merged_score.csv")
    write_maxima(maxima, original_data, header)
    return statistics, merged


if __name__ == '__main__':
    score_paraphrasations()
