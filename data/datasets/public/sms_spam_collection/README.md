# SMS Spam Collection v1

Source: https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection
Licence: CC BY 4.0

Citation
--------
Almeida, A., Gomez Hidalgo, J.M., Yamakami, A. (2011). Contributions to the study of SMS spam filtering. Proceedings of the 2011 European Conference on Machine Learning and Principles of Data Mining (ECML PKDD 2011), LNCS 6913, pp. 307-314.

How DigitalMind uses it
-----------------------
The multi-label risk indicators (urgency, threat language, emotional pressure, payment
request) are projected onto this real text with DigitalMind's own rule lexicon
(`app/datasets.py:_spans_for_labels`). The projection is auditable and printed in
`reports/evaluation_report.md`; it is NOT ground truth for those indicators. What the
dataset provides that our synthetic corpus cannot is *real human wording* for pressure
tactics, so it tests whether the classifier generalises beyond its own templates.

This directory is not committed. Run:

    python backend/scripts/fetch_public_data.py
