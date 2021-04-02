import csv
import argparse
from uuid import UUID
from typing import Optional, Dict

from utils import is_parsable


DEFAULT_QUOTING: int = csv.QUOTE_ALL
QUOTING_RULES: Dict[str, int] = {key: getattr(csv, key)
                                 for key in csv.__dict__.keys()
                                 if key.startswith("QUOTE_")}


def validate_cleanup(original_filename: str, modified_filename: str,
                     quoting: int = DEFAULT_QUOTING,
                     overwrite: Optional[bool] = None):
    with open(original_filename) as original_file:
        original = list(csv.reader(original_file))
    with open(modified_filename) as modified_file:
        modified = list(csv.reader(modified_file))
        if len(modified[0]) < len(original[0]):
            modified_file.seek(0)
            modified = list(csv.reader(modified_file, delimiter=";"))
    if not original[0] == modified[0]:
        print(f"faulty header detected in {modified_filename}")
    out = [original[0]]
    for row in original[1:]:
        assert is_parsable(row[0], UUID)
        for mod_row in modified:
            if mod_row[0] == row[0]:
                assert len(mod_row) == len(row)
                for i in range(len(row)):
                    assert is_parsable(row[i], int) \
                           == is_parsable(mod_row[i], int)
                out.append(mod_row)
                break
    print(f"""
{len(out[1:]):5d} stories to write from and to {modified_filename}
{(len(modified[1:]) - len(out[1:])):5d} invalid stories or duplicates to delete
{len(original[1:]):5d} stories in the original file {original_filename}
quoting settings are {[k for k, v in QUOTING_RULES.items() if v == quoting][0]}
    """)
    if overwrite is None:
        answer = input(f"Overwrite {modified_filename}? [y/N] ")
    else:
        answer = "yes" if overwrite is True else "no"
    if answer.lower().startswith("y"):
        print(f"Overwriting {modified_filename} ...")
        with open(modified_filename, "w") as out_file:
            writer = csv.writer(out_file, quoting=quoting)
            for row in out:
                writer.writerow(row)
    else:
        print("Nothing written")


class DictAction(argparse.Action):
    def __call__(self, parser: argparse.ArgumentParser,
                 namespace: argparse.Namespace, value: str,
                 option_string: Optional[str] = None) -> None:
        setattr(namespace, self.dest, QUOTING_RULES.get(value))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", default="cloze_test.csv",
                        help="the original dataset file in the right format")
    parser.add_argument("modified", help="the file to validate and overwrite")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--yes-overwrite", dest="overwrite", default=None,
                       action="store_true", help="overwrite MODIFIED")
    group.add_argument("--no-overwrite", dest="overwrite", default=None,
                       action="store_false", help="do not overwrite MODIFIED")
    parser.add_argument("--quoting", default=DEFAULT_QUOTING,
                        action=DictAction, choices=list(QUOTING_RULES.keys()),
                        help="csv quoting settings to apply when writing")
    args = parser.parse_args()
    validate_cleanup(args.original, args.modified, overwrite=args.overwrite)


if __name__ == '__main__':
    main()
