"""The published dashboard's data: ``python -m assetforecast.dashboard`` writes ``dashboard/data/*.csv``.

* ``data``      one builder per dataset, the dataset registry and the manifest (``dashboard/datasets.json``)
* ``verdicts``  the rules that pick which of the page's prewritten sentences applies
* ``glossary``  plain-language names of models, rules, portfolios and assets
* ``page``      rebuilds ``dashboard/index.html`` as one self-contained file with the data inlined
* ``summary``   writes ``dashboard/summary.html``, the one-page plain-language summary (a printable sheet)

The page itself is ``dashboard/index.html``; open it in any browser. ``dashboard/summary.html`` is its
one-sheet companion, generated whole by ``summary`` (nothing in it is hand-written).
Its hand-written part, between ``<!-- page -->`` and ``<!-- /page -->``, binds every
figure to a dataset by ``data-source`` / ``data-field`` / ``data-where`` and holds every sentence.
"""
