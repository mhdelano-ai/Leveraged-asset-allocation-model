"""Walk-forward engine, scoring and tests for the supervised mean models.

This engine (purged training windows, refit schedule, forecast cache) serves the
return side and the replication. The covariance models use ``assetforecast.eval``.
"""
