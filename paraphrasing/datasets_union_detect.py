import csv

found = 0
not_found = 0

with open('../cloze_test_nolabel.csv') as cloze_test_nolabel_file:
    cloze_test_nolabel = list(csv.DictReader(cloze_test_nolabel_file))
with open('../cloze_test.csv') as cloze_test_file:
    cloze_test = list(csv.DictReader(cloze_test_file))

cols = [
    "InputStoryid",
    "InputSentence1", "InputSentence2", "InputSentence3", "InputSentence4",
    "RandomFifthSentenceQuiz1", "RandomFifthSentenceQuiz2",
    "AnswerRightEnding"  # only for cloze_test
]

right_ending = [0, 0]

for row_nolabel in cloze_test_nolabel:
    for row_label in cloze_test:
        if all(row_nolabel[col] == row_label[col]
               for col in cols[1:-1]
               # if col.startswith("InputSentence")
               ):
            found += 1
            right_ending[int(row_label["AnswerRightEnding"]) - 1] += 1
            break
    else:
        not_found += 1

print("Identical stories:", found, "out of", found + not_found)
print("Right endings:", dict(enumerate(right_ending)))
