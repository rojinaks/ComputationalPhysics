import numpy as np


def normalized_autocorrelation(
    time_series: np.ndarray, mean: float | None = None
) -> np.ndarray:

    if mean is None:
        mean = time_series.mean()

    series: np.ndarray = time_series - mean

    C: np.ndarray = np.zeros_like(time_series)

    n = len(series)
    for t in range(n - 1):
        C[t] = (series[: n - t] * series[t:]).mean()

    return C / C[0]


def integrated_autocorrelation(correlation: np.ndarray):
    # until sign flip

    # for i in range(2, len(correlation)):
    #     tau += correlation[i]

    #     if correlation[i] * correlation[i - 1] < 0:
    #         break

    # return tau

    try:
        first_zero = np.where(correlation <= 0)[0][0]
        return 0.5 + correlation[1:first_zero].sum()
    except:
        return len(correlation)  # at least!
