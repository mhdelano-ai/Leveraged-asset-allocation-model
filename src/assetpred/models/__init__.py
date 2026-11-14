"""Forecasting models.

* ``base``      the :class:`~assetpred.models.base.Forecaster` interface and helpers
* ``registry``  name -> factory registry (``register``, ``create``, ``available``)
* ``baselines`` benchmarks: ``hist_mean`` (THE benchmark), ``hist_quantile``,
                ``zero``, ``vol_ewma`` (volatility-aware benchmark), ``vol_ewma_fhs``
* ``linear``    ``ridge``: pooled ridge, a reference point model and template

Any module added here that calls ``registry.register`` is picked up
automatically by the evaluation runner.
"""
