<!-- omit in toc -->
# Data Curation: Filtering and Deduplication

At this scale, what you train on matters more than almost any architecture choice. FineWeb, DCLM and RefinedWeb all showed the same thing: the same model trained on cleaned web text beats one trained on raw web text, often by a lot. Cleaning comes in two parts, both in [`src/curation/`](https://github.com/ghimiresunil/building-llms-from-scratch/tree/main/src/curation).

## Quality filters

Web text is full of navigation menus, cookie banners, keyword spam, tables of numbers, and pages that say "Read more..." twenty times. A few cheap rules, adapted from Gopher and C4, catch most of it. Each rule looks at one symptom:

| Rule | Fires when | Catches |
|---|---|---|
| `too_short` | fewer than 50 words | fragments, error pages |
| `word_length` | mean word length outside 3 to 10 | number tables, URLs, minified code |
| `symbols` | too many `#` or `...` per word | hashtag spam, teasers |
| `non_alpha` | under 80% of words contain a letter | numeric dumps |
| `ellipsis_lines` | over 30% of lines end in `...` | lists of article teasers |
| `bullet_lines` | over 90% of lines are bullets | menus |
| `duplicate_lines` | over 30% of lines are repeats | boilerplate, footers |
| `repetition` | one word pair covers over 20% of the text | keyword spam |
| `banned_phrase` | "lorem ipsum" and friends | placeholder pages |

All the thresholds are fields of `QualityConfig`. They're deliberately loose: a filter that throws away good text costs you data you can't get back.

## Deduplication

The web repeats itself: syndicated news, licenses, templated pages. Training on duplicates wastes compute and pushes the model toward memorizing (Lee et al., 2022).

**Exact duplicates** are easy: hash the text after lowercasing and stripping punctuation.

**Near duplicates** need a similarity measure. The usual one is the Jaccard similarity of each document's set of word 5-grams ("shingles"). Comparing all pairs is quadratic, so two tricks make it linear:

1. **MinHash.** Hash every shingle with 128 different hash functions and keep the minimum of each. The probability that two documents share a minimum equals their Jaccard similarity, so the fraction of matching signature entries estimates it.
2. **LSH banding.** Split the 128 numbers into 16 bands of 8. Two documents become candidates only if they match on every number in at least one band. Similar pairs almost always collide somewhere; dissimilar ones almost never do. Candidates are then checked against the real threshold (0.8 by default).

```python
dedup = Deduplicator(threshold=0.8, num_perm=128, bands=16)
for doc in documents:
    if not quality_issues(doc) and not dedup.is_duplicate(doc):
        keep(doc)
```

## Running it

```bash
python scripts/curate_data.py --input data/raw/train --limit 20000 --dry_run      # statistics only
python scripts/curate_data.py --input data/raw/train --out_dir data/curated/train \
    --rejects logs/rejects.jsonl                                                   # keep what was dropped, to inspect
```

It reads `.jsonl` or `.jsonl.zst` files with a `text` field, the format The Pile uses, and writes `curated.jsonl.zst` in the same format, so `scripts/data_preprocess.py` can tokenize the folder as usual. Read a sample of the rejects before trusting the numbers: every filter has false positives, and you should know what yours are.

A good experiment: pretrain the same small model on raw and on curated data for the same number of tokens, and compare the dev loss on a *held-out curated* set.

## References

- Rae et al. (2021). *Scaling Language Models: Methods, Analysis & Insights from Training Gopher* (quality rules).
- Raffel et al. (2020). *Exploring the Limits of Transfer Learning* (C4 filters).
- Broder (1997). *On the Resemblance and Containment of Documents* (MinHash).
- Lee et al. (2022). *Deduplicating Training Data Makes Language Models Better.*
- Penedo et al. (2024). *The FineWeb Datasets.*
