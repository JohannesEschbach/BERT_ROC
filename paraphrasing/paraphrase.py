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
            csv_in_reader = csv.reader(csv_in)
            missing_rows = []
            if os.path.isfile(current_dest):
                print("completing existing file:", current_dest)
            else:
                with open(current_dest, "a") as csv_dest:
                    csv_dest.write(csv_in.readline())
            with open(current_dest) as csv_dest:
                csv_dest_reader = csv.reader(csv_dest)
                for row in csv_in_reader:
                    csv_dest.seek(0)
                    if not len(row):
                        continue
                    uuid = row[0]
                    try:
                        UUID(uuid)
                    except ValueError:
                        continue
                    for row_dest in csv_dest_reader:
                        if uuid == row_dest[0]:
                            break
                    else:
                        missing_rows.append(row)
            print("Translating", current_src, "to", current_dest)
            print(f"{len(missing_rows)} missing rows")
            rows = translator.translate(lang, missing_rows)
            with open(current_dest, "a") as csv_out:
                csv_writer = csv.writer(csv_out)
                for row in rows:
                    csv_writer.writerow(row)
        current_src = current_dest


def main(dataset: str = "negated_synonymized",
         translator: Translator = DeepLTranslator(),  # GoogleTranslator(),
         wd: str = os.path.realpath(os.path.join(os.getcwd(), "data")),
         lang_input: str = "en"):
    fixed_hops = ["ja-JA"]
    fixed_hops = [[lang
                   for lang in translator.dest_languages
                   if lang.startswith(hop)][0]
                  for hop in fixed_hops]
    lang_target = [lang
                   for lang in translator.dest_languages
                   if lang.startswith(lang_input)][0]
    lang_pipelines = [fixed_hops + [middle, lang_target]
                      for middle in translator.dest_languages
                      if middle != lang_target
                      and not (len(fixed_hops) and middle == fixed_hops[-1])
                      and not middle.startswith(lang_input)]
    for languages in reversed(lang_pipelines):
        paraphrase(dataset, languages, wd, translator)


if __name__ == "__main__":
    main()
