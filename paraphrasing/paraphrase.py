import os
import csv
from uuid import UUID
from typing import List

from translate import Translator, DeepLTranslator, GoogleTranslator


def paraphrase(src: str, languages: List[str], wd: str,
               translator: Translator):
    wd = os.path.realpath(wd)
    if src.endswith(".csv"):
        src = src[:-4]
    current_src = os.path.join(wd, f"{src}.csv")
    for i, lang in enumerate(languages):
        if not os.path.isfile(current_src):
            raise FileNotFoundError(current_src)
        filename = f"{src}.{'_'.join(languages[:i+1])}.csv"
        current_dest = os.path.join(wd, filename)
        with open(current_src) as csv_in:
            if os.path.isfile(current_dest):
                print("completing existing file:", current_dest)
            else:
                with open(current_dest, "a") as csv_dest:
                    csv_dest.write(current_dest.readline())
            print("Translating", current_src, "to", current_dest)
            csv_in_reader = csv.reader(csv_in)
            missing_rows = []
            with open(current_dest) as csv_dest:
                csv_dest_reader = csv.reader(csv_dest)
                for row in csv_in_reader:
                    uuid = row[0]
                    try:
                        UUID(uuid)
                    except ValueError:
                        continue
                    for row_dest in csv_dest_reader:
                        if uuid == row[0]:
                            break
                    else:
                        missing_rows.append(row)
            print(f"{len(missing_rows)} missing rows")
            rows = translator.translate(lang, missing_rows)
            with open(current_dest, "a") as csv_out:
                csv_writer = csv.writer(csv_out)
                for row in rows:
                    csv_writer.writerow(row)
        current_src = current_dest


if __name__ == "__main__":
    wd = os.path.realpath(os.path.join(os.getcwd(), "data"))
    lang_input = "en"
    translator = GoogleTranslator()
    translator = DeepLTranslator(headless=True)
    lang_target = [lang
                   for lang in translator.dest_languages
                   if lang.startswith(lang_input)][0]
    lang_pipelines = [[middle, lang_target]
                      for middle in translator.dest_languages
                      if not middle.startswith(lang_input)]
    datasets = [
        "cloze_test",
        # "cloze_test_nolabel",
    ]
    for dataset in datasets:
        for languages in reversed(lang_pipelines):
            paraphrase(dataset, languages, wd, translator)
