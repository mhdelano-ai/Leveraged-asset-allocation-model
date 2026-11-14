# condcorr — correlations conditioned on economic state

Package `src/volmodel/models/condcorr/`, tests `tests/models/test_condcorr.py`.
All numbers below are from the development sample (`VOLMODEL_MAX_DATE=2014-12-31`,
targets to origin 2013-12); nothing after 2014 was loaded.

## The question

Can observable conditions tell us which stock–bond correlation regime the next
12 months will be in, faster and for a better reason than a slow average? The
stock–bond correlation is the one number in the matrix that has changed sign
(realized 12-month us_large–ust_int correlation: about −0.2..+0.1 in 1955–65,
+0.3..+0.7 in 1968–1996, −0.7..+0.1 from 2000). Every model in the family
leaves the rest of the matrix at a sound unconditional estimate and moves only
a block.

__RESULTS__
