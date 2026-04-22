# Compression Similarity

This research article asks if sequence representation or program representation gives better distance metric to capture perceived music similarity. Here we focus on symbolic music that are sequentially organized and can be derived under a musical grammar.

One interesting distance metric is **compression distance**.

- Compression distance between x and y in the most general form is defined as the length of the shortest program that transform x to y. However, this value is intractable to compute.
- In practice [citation], it is approximated with "normalized compression distance (NCD)". Given a function $C(x|y)$ representing the encoding size of $x$ using information about $y$, the normalized compression distance is defined as
$$NCD(x,y) := \frac{max(C(x|y),C(y|x))}{max(C(x),C(y))}$$
- In this article, we compare two choices of the function $C$, one is based on a sequential model (PPM), the other is based on library learning of programs.
  - In the PPM case, $C_{ppm}(x|y) := \sum_{x_i \in x} h_{m_y}(x_i)$
    - "aggregated surprisal on $x$ for a prediction model trained on $y$"
  - In the Stitch case, $C_{stitch}(x|y) := size(rewrite(x,pat(y)))$
    - "size of x refactored using patterns in y (counting definition size of the used patterns)"

## Set-up

install Python packages

``` zsh
pip install -r requirements.txt
```

To reproduce the results, do

``` zsh
python src/main.py
```

This will update the `data/result` folder with newly produced data results, which in automatically updates the figures in `doc/figs`.

## Directory structure

`src` contains all the codes related to experiment, core model, and preprocessing.

`data` contains dataset and result data (JSON only)

`doc` contains all the documents (in sync with results from `data`)

### high-level dependency structure

- `src` read and write to `data`
- `doc` read from `data`

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
