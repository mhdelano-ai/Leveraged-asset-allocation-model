"""Joint predictive distribution: forecasts of the covariance matrix of h-month
excess log returns, their out-of-sample evaluation, and a combinator with the
marginal models' sds.

Modules: ``base`` (interface, information set, registry), ``estimators`` and
``linalg`` (building blocks), ``models`` (registered models, ``FROZEN``),
``engine`` (walk-forward, cache), ``scoring``, ``combine``, ``run`` (CLI).
"""
