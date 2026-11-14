"""The published dashboard's data: ``python -m assetforecast.dashboard`` writes ``dashboard/data/*.csv``.

* ``data``      one builder per dataset, the dataset registry and the manifest (``dashboard/datasets.json``)
* ``verdicts``  the rules that pick which of the page's prewritten sentences applies
* ``glossary``  plain-language names of models, rules, portfolios and assets
* ``page``      rebuilds ``dashboard/index.html`` as one self-contained file with the data inlined

The page itself is ``dashboard/index.html``, the repo's only page; open it in any browser.
Its hand-written part, between ``<!-- page -->`` and ``<!-- /page -->``, binds every
figure to a dataset by ``data-source`` / ``data-field`` / ``data-where`` and holds every sentence.
"""
