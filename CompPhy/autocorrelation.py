import numpy as np
from numpy.typing import NDArray


def normalized_autocorrelation(time_series: np.ndarray, mean: float | None = None, fast: bool = True) -> np.ndarray:

    if mean is None:
        mean = time_series.mean()

    series: np.ndarray = time_series - mean

    if not fast:
        C: np.ndarray = np.zeros_like(time_series)

        n = len(series)
        for t in range(n - 1):
            # C[t] = (series[: n - t] * series[t:]).mean()
            C[t] = (series * np.roll(series, shift=t, axis=0)).mean()
    else:
        C = np.fft.ifft(np.fft.fft(time_series) * np.fft.ifft(time_series)).real

    return C / C[0]


def integrated_autocorrelation(time_series: np.ndarray, mean: float | None = None, fast: bool = True) -> int:
    # until sign flip

    # for i in range(2, len(correlation)):
    #     tau += correlation[i]

    #     if correlation[i] * correlation[i - 1] < 0:
    #         break

    # return tau

    C = normalized_autocorrelation(time_series, mean, fast)

    try:
        first_zero = np.where(C <= 0)[0][0]
        return int(np.ceil(0.5 + C[1:first_zero].sum()))
    except:
        return len(C)  # at least!


def blocking(time_series: np.ndarray, mean: float | None = None, tau: float | None = None, fast: bool = True):
    if tau is None:
        tau = int(np.ceil(integrated_autocorrelation(time_series, mean, fast)))

    n = len(time_series) // tau

    blocked = np.zeros(n)

    for b in range(n):
        blocked[b] = time_series[b * tau : (b + 1) * tau].mean()

    return blocked


def mean_spin_spin_correlation(cfgs: NDArray[np.int8]) -> NDArray[np.float64]:
    """

    Calculate the spin-spin two-point correlation function using the fast
    fourier transform algorithm for a set of spin configurations

    """

    return np.fft.ifft2(np.fft.fft2(cfgs, axes=(1, 2)) * np.fft.ifft2(cfgs, axes=(1, 2)), axes=(1, 2)).real


__all__ = ["normalized_autocorrelation", "integrated_autocorrelation", "blocking", "mean_spin_spin_correlation"]
