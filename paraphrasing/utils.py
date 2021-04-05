from typing import Iterator, Iterable, List, Sequence, Any, Generator, Dict
import numpy as np


class Buckets(Iterator):
    """
    Get an Iterator from an Iterable, yielding lists of length up to the
    bucket_size specified containing the elements in their respective order.
    """
    content: Iterable
    bucket_size: int

    def __init__(self, content: Iterable, bucket_size: int):
        if not bucket_size > 0:
            raise ValueError("bucket_size must be greater than zero")
        self.content = content
        self.bucket_size = bucket_size

    def __iter__(self):
        self.content = iter(self.content)
        return self

    def __next__(self):
        cumulator = []
        try:
            cumulator.append(next(self.content))
        except StopIteration:
            raise StopIteration
        for i in range(self.bucket_size - 1):
            try:
                cumulator.append(next(self.content))
            except StopIteration:
                break
        return cumulator


def minimum_edit_distance(a: str, b: str,
                          insert: int = 1, delete: int = 1, update: int = 2) \
 -> int:
    """
    Compute the minimum required number of insert, delete or update operations,
    weighted by the respective parameters.
    """
    distances = np.zeros((len(a) + 1, len(b) + 1), dtype=int)
    for i in range(len(a)):
        distances[i, 0] = i
        for j in range(len(b)):
            distances[0, j] = j
            if a[i] == b[j]:
                distances[i+1, j+1] = distances[i, j]
            else:
                distances[i+1, j+1] = min(insert + distances[i+1, j],
                                          delete + distances[i, j+1],
                                          update + distances[i, j])
    return distances[len(a), len(b)]


def bleu_score(translation: List[str],  # already tokenised
               gold_standards: List[List[str]],  # list of tokenised texts
               n_gram: int = 1) -> float:
    length: int = len(translation)
    score: float = 0
    for subsequence in unique(subsequences(translation, n_gram)):
        freq_max_gold = max(count_subsequences(subsequence, gold_standard)
                            for gold_standard in gold_standards)
        freq_translation = count_subsequences(subsequence, translation)
        if freq_translation > freq_max_gold:
            freq_translation = freq_max_gold
        score += freq_translation/length


def unique(gen: Generator[Any, None, None]) -> Generator[Any, None, None]:
    processed = []
    for item in gen:
        if item not in processed:
            processed.append(item)
            yield item


def subsequences(seq: Sequence[Any],
                 length: int) -> Generator[Sequence[Any], None, None]:
    return (seq[i:i+length] for i in range(len(seq) - length + 1))


def count_subsequences(needle: List[Any], haystack: List[Any]) -> int:
    return sum(any(h == n for h in haystack) for n in needle)


def is_parsable(text: str, type_func: type) -> bool:
    try:
        type_func(text)
        return True
    except ValueError:
        pass
    return False
