This repository contains a Named Entity Recognition (NER) model trained on a modified CoNLL format dataset for detecting time expressions.
The model follows a BIO tagging scheme with the following labels:

B-T → Beginning of a Time Expression

I-T → Inside a Time Expression

O → Outside (non-time tokens)

PAD → Padding (for batching only)

The goal is to accurately identify spans of temporal expressions (e.g., dates, times, durations) in text.
