# Compression Similarity


## Overview


## Data Format

Data is represented as transitions between chords, where a transition string has 3 parts: the interval between the roots of the two chords (in ascending semitones), the quality of the first chord, the quality of the second chord.

For example:

- (G7, Cmaj7) -> "7,7,maj7"
- (Dm, G7) -> "7,m,7"
- (Am, C) -> "3,m,maj"


## Conversion

Run the conversion:
```bash
python data_formatting.py
```

This will:
1. Read all entries from `treebank.json`
2. Convert each entry using `convert_treebank_to_corpus_rep()`
3. Filter out any entries without valid tree structures
4. Write the results to `converted_corpus_rep.json`
