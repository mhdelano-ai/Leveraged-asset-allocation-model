"""Return forecasting models (mean models for the replication, and benchmarks).

* ``base``      the :class:`~assetforecast.returns.models.base.Forecaster` interface and helpers
* ``registry``  name -> factory registry (``register``, ``create``, ``available``)
* ``baselines`` benchmarks: ``hist_mean`` (THE benchmark), ``hist_quantile``,
                ``zero``, ``vol_ewma`` (volatility-aware benchmark), ``vol_ewma_fhs``
* ``ml_mean``   the frozen standardized-deviation mean models (``ridge_z`` ...)

Any module added here that calls ``registry.register`` is picked up
automatically by the evaluation runner. These are kept apart from
``assetforecast.models`` (the covariance models), whose discovery imports every
subpackage there.
"""
