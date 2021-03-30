from typing import Iterator, Iterable
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
    dst = np.zeros((len(a) + 1, len(b) + 1), dtype=int)
    for i in range(len(a)):
        dst[i, 0] = i
        for j in range(len(b)):
            dst[0, j] = j
            if a[i] == b[j]:
                dst[i+1, j+1] = dst[i, j]
            else:
                dst[i+1, j+1] = min(insert + dst[i+1, j],
                                    delete + dst[i, j+1],
                                    update + dst[i, j])
    return dst[len(a), len(b)]
