from typing import Iterator, Iterable


class Buckets(Iterator):
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
