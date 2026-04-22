import json
from itertools import chain

with open("../../../data/corpus/preprocessed/converted_corpus_rep.json", "r") as f:
    data = json.load(f)

progs = [x["progression"].split() for x in data]
symbols = set(chain(*progs))

print(len(set(symbols)))
