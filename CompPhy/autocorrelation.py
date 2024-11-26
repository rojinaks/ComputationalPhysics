import numpy as np


def normalized_autocorrelation(time_series: np.ndarray, mean: float | None = None) -> np.ndarray:

    if mean is None:
        mean = time_series.mean()

    series: np.ndarray = time_series - mean

    C: np.ndarray = np.zeros_like(time_series)

    n = len(series)
    for t in range(n - 1):
        C[t] = (series[: n - t] * series[t:]).mean()

    return C / C[0]


def integrated_autocorrelation(time_series: np.ndarray, mean: float | None = None):
    # until sign flip

    # for i in range(2, len(correlation)):
    #     tau += correlation[i]

    #     if correlation[i] * correlation[i - 1] < 0:
    #         break

    # return tau

    C = normalized_autocorrelation(time_series, mean)

    try:
        first_zero = np.where(C <= 0)[0][0]
        return 0.5 + C[1:first_zero].sum()
    except:
        return len(C)  # at least!


def blocking(time_series: np.ndarray, mean: float | None = None, tau: float | None = None):
    if tau is None:
        tau = int(np.ceil(integrated_autocorrelation(time_series, mean)))

    n = len(time_series) // tau

    blocked = np.zeros(n)

    for b in range(n):
        blocked[b] = time_series[b * tau : (b + 1) * tau].mean()

    return blocked


__all__ = ["normalized_autocorrelation", "integrated_autocorrelation", "blocking"]
