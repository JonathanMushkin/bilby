"""
Marginalization of the likelihood over the extrinsic parameters (sky
location, arrival time, polarization, orbital phase and distance) from
matched-filter time series, following Roulet, Mushkin et al.,
arXiv:2404.02435, and its implementation in cogwheel.

Sky locations are given by Earth-fixed latitude and longitude, at one
fixed GPS time: the rotation of the Earth during the signal is neglected.
At that time the latitude equals the declination and the longitude is the
right ascension minus the Greenwich mean sidereal time.
"""

import itertools
from dataclasses import dataclass, field

import numpy as np
import scipy.signal
from scipy.interpolate import krogh_interpolate, make_interp_spline
from scipy.special import logsumexp
from scipy.stats import cauchy, qmc

from ...core.utils import logger, speed_of_light
from ..geometry import get_polarization_tensor
from ..time import greenwich_mean_sidereal_time


def get_antenna_patterns_at_zero_polarization(
    interferometers, latitude, longitude, time
):
    """
    Return the plus and cross antenna patterns at polarization zero.

    Parameters
    ----------
    interferometers: bilby.gw.detector.InterferometerList
        The interferometers.
    latitude, longitude: array_like
        Earth-fixed latitude and longitude of the source (rad), of equal
        shape.
    time: float
        GPS time at which the sky location is converted to right ascension.

    Returns
    -------
    antenna_patterns: array_like
        Of shape (..., number of interferometers, 2), plus first, where
        ... is the shape of :code:`latitude`.
    """
    right_ascension = np.asarray(longitude) + greenwich_mean_sidereal_time(time)
    polarization_tensors = [
        get_polarization_tensor(
            np.ravel(right_ascension), np.ravel(latitude), time, 0.0, mode
        )
        for mode in ["plus", "cross"]
    ]
    antenna_patterns = np.array(
        [
            [
                np.einsum(
                    "ij,ij...->...", interferometer.geometry.detector_tensor, tensor
                )
                for tensor in polarization_tensors
            ]
            for interferometer in interferometers
        ]
    )
    return np.moveaxis(antenna_patterns, -1, 0).reshape(
        np.shape(latitude) + antenna_patterns.shape[:2]
    )


def get_geocenter_delays(interferometers, latitude, longitude):
    """
    Return the arrival time at each interferometer minus the arrival time
    at the geocenter.

    Parameters
    ----------
    interferometers: bilby.gw.detector.InterferometerList
        The interferometers.
    latitude, longitude: array_like
        Earth-fixed latitude and longitude of the source (rad), of equal
        shape.

    Returns
    -------
    delays: array_like
        Of shape (number of interferometers, ...), in seconds, where ... is
        the shape of :code:`latitude`.
    """
    latitude, longitude = np.asarray(latitude), np.asarray(longitude)
    direction = np.array(
        [
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        ]
    )
    vertices = np.array([interferometer.vertex for interferometer in interferometers])
    return -np.einsum("di,i...->d...", vertices, direction) / speed_of_light


def rotate_antenna_patterns(antenna_patterns, psi):
    """
    Return the antenna patterns at polarization :code:`psi`, given those
    at polarization zero.

    Parameters
    ----------
    antenna_patterns: array_like
        Of shape (..., number of interferometers, 2), plus first, at
        polarization zero.
    psi: array_like
        Polarization (rad), of shape (...).

    Returns
    -------
    antenna_patterns: array_like
        Of the same shape as the input, at polarization :code:`psi`.
    """
    cos_two_psi = np.cos(2 * np.asarray(psi))[..., np.newaxis]
    sin_two_psi = np.sin(2 * np.asarray(psi))[..., np.newaxis]
    plus, cross = antenna_patterns[..., 0], antenna_patterns[..., 1]
    return np.stack(
        [
            cos_two_psi * plus + sin_two_psi * cross,
            -sin_two_psi * plus + cos_two_psi * cross,
        ],
        axis=-1,
    )


def exp_normalize(log_probability, axis=-1):
    """
    Return probabilities normalized to sum to one along :code:`axis`, from
    unnormalized log probabilities, as cogwheel's :code:`utils.exp_normalize`.
    """
    probability = np.exp(
        log_probability - np.max(log_probability, axis=axis, keepdims=True)
    )
    return probability / np.sum(probability, axis=axis, keepdims=True)


class SkyDictionary:
    """
    Sky locations isotropic in Earth-fixed latitude and longitude, grouped
    by their arrival-time delays between interferometers, following
    arXiv:2404.02435 Sec. II D 1 and cogwheel's :code:`SkyDictionary`.

    The delays from the first interferometer to the others are rounded to
    multiples of :code:`1 / f_sampling`; each tuple of rounded delays is a
    cell, and :code:`delays2inds_map` maps it to the indices of the sky
    locations in it.

    Parameters
    ----------
    interferometers: bilby.gw.detector.InterferometerList
        The interferometers; the first is the reference for the delays.
    f_sampling: int, optional
        Inverse of the width of a delay cell (Hz). Default is :code:`2 ** 13`.
    nsky: int, optional
        Number of sky locations. Default is :code:`10 ** 6`.
    seed: int, optional
        Seed of the quasi-random sequence of sky locations.

    Attributes
    ----------
    sky_samples: dict
        :code:`lat` and :code:`lon`, Earth-fixed latitude and longitude (rad).
    fplus_fcross_0: array_like
        Antenna patterns at polarization zero, of shape (nsky, number of
        interferometers, 2), plus first.
    geocenter_delay_first_det: array_like
        Arrival time at the first interferometer minus that at the geocenter
        (s), of shape (nsky,).
    delays: array_like
        Arrival time at each other interferometer minus that at the first
        (s), of shape (number of interferometers - 1, nsky).
    delays2inds_map: dict
        Maps each cell, a tuple of rounded delays in units of
        :code:`1 / f_sampling`, to an array of sky indices.
    """

    DEFAULT_F_SAMPLING = 2**13

    def __init__(
        self, interferometers, *, f_sampling=DEFAULT_F_SAMPLING, nsky=10**6, seed=0
    ):
        self.detector_names = tuple(
            interferometer.name for interferometer in interferometers
        )
        self.nsky = nsky
        self.f_sampling = f_sampling
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        self.sky_samples = self._create_sky_samples()
        # In Earth-fixed coordinates the antenna patterns do not depend on time; any GPS time gives the same values
        self.fplus_fcross_0 = get_antenna_patterns_at_zero_polarization(
            interferometers, self.sky_samples["lat"], self.sky_samples["lon"], 0.0
        ).astype(np.float32)
        geocenter_delays = get_geocenter_delays(
            interferometers, self.sky_samples["lat"], self.sky_samples["lon"]
        )
        self.geocenter_delay_first_det = geocenter_delays[0]
        self.delays = geocenter_delays[1:] - geocenter_delays[0]

        self.delays2inds_map = self._create_delays2inds_map()
        if len(self.detector_names) == 1:
            self._min_delay = np.array([], dtype=int)
            self._max_delay = np.array([], dtype=int)
        else:
            discrete_delays = np.array(list(self.delays2inds_map))
            self._min_delay = np.min(discrete_delays, axis=0)
            self._max_delay = np.max(discrete_delays, axis=0)

        # Priors of the rounded delays between ordered sets of 2 or 3 interferometers, weighted by |F|^3, for the
        # proposal of arrival times (apply_tdet_prior)
        self._delays_bounds = dict()
        self._delays_prior = dict()
        weights = np.linalg.norm(self.fplus_fcross_0, axis=(1, 2)) ** 3
        for number in range(2, len(self.detector_names) + 1):
            for order in itertools.permutations(
                range(len(self.detector_names)), number
            ):
                pairwise_delays = self._discretize(
                    geocenter_delays[order[1:],] - geocenter_delays[order[0]]
                )
                bins = [
                    np.arange(*bounds) + 0.5
                    for bounds in zip(
                        pairwise_delays.min(axis=1) - 1, pairwise_delays.max(axis=1) + 1
                    )
                ]
                self._delays_prior[order] = np.histogramdd(
                    pairwise_delays.T, weights=weights, bins=bins
                )[0].copy("F")
                if number == 2:
                    self._delays_bounds[order] = (
                        pairwise_delays.min(),
                        pairwise_delays.max(),
                    )

        # An array over cells of iterators that cycle through each cell's sky indices
        self._ind_generators = np.full(self._max_delay - self._min_delay + 1, iter(()))
        self.set_generators()

        # The fraction of sky locations in each cell times the cell width, 1 / f_sampling: the factor Delta Pi(delta
        # tau) of arXiv:2404.02435 Eq. 28. Negative rounded delays index the array from its end, as in cogwheel; this
        # works because the array spans max - min + 1 cells in each dimension
        self._sky_prior = np.zeros(self._max_delay - self._min_delay + 1)
        for key, inds in self.delays2inds_map.items():
            self._sky_prior[key] = len(inds) / self.nsky / self.f_sampling

    def set_generators(self):
        """
        Reset the iterators over each cell's sky indices, so that
        :code:`get_sky_inds_and_prior` repeats its output.
        """
        for key, inds in self.delays2inds_map.items():
            self._ind_generators[key] = itertools.cycle(inds)

    def resample_timeseries(self, timeseries, times, axis=-1):
        """
        Resample a time series to :code:`f_sampling`, after a square-root
        Tukey taper of its edges towards the mean of its two end values, as
        cogwheel's :code:`SkyDictionary.resample_timeseries`.

        Parameters
        ----------
        timeseries: array_like
            The time series.
        times: array_like
            Its equally spaced times (s); their spacing times
            :code:`f_sampling` must be an integer.
        axis: int, optional
            The time axis of :code:`timeseries`.

        Returns
        -------
        timeseries, times: array_like
            The resampled time series and its times.
        """
        shape = [1 for _ in timeseries.shape]
        shape[axis] = timeseries.shape[axis]
        window = np.sqrt(scipy.signal.windows.tukey(shape[axis], 0.1)).reshape(shape)
        endpoint = np.mean(timeseries.take([0, -1], axis), axis, keepdims=True)
        timeseries = (timeseries - endpoint) * window + endpoint

        ratio = np.round(self.f_sampling * (times[1] - times[0]), decimals=10)
        if ratio != 1:
            timeseries, times = scipy.signal.resample(
                timeseries, int(len(times) * ratio), times, axis=axis
            )
            if not np.isclose(1 / self.f_sampling, times[1] - times[0]):
                raise ValueError(
                    "The spacing of `times` is incommensurate with `f_sampling`."
                )
        return timeseries, times

    def get_sky_inds_and_prior(self, delays):
        """
        Return a sky location for each tuple of rounded delays, and the
        prior factor of its cell.

        Parameters
        ----------
        delays: array_like
            Integers, of shape (number of interferometers - 1, number of
            samples): delays from the first interferometer, in units of
            :code:`1 / f_sampling`.

        Returns
        -------
        sky_inds: array_like
            Sky indices, one for each physical sample, taken in turn from
            the cell's sky locations.
        sky_prior: array_like
            The cell's fraction of sky locations times :code:`1 / f_sampling`
            (s), for each physical sample.
        physical_mask: array_like
            Boolean, of length number of samples: whether any sky location
            has these delays.
        """
        if len(self.detector_names) == 1:
            number = delays.shape[1]
            sky_inds = np.fromiter(
                itertools.islice(self._ind_generators[()], number), int, number
            )
            return sky_inds, np.full(number, self._sky_prior), np.full(number, True)

        physical_mask = np.all(
            (delays.T >= self._min_delay) & (delays.T <= self._max_delay), axis=1
        )
        sky_prior = self._sky_prior[tuple(delays[:, physical_mask])]
        submask = sky_prior > 0
        physical_mask[physical_mask] *= submask
        sky_prior = sky_prior[submask]
        generators = self._ind_generators[tuple(delays[:, physical_mask])]
        sky_inds = np.fromiter(map(next, generators), int, len(generators))
        return sky_inds, sky_prior, physical_mask

    @classmethod
    def choose_f_sampling(cls, f_nyquist):
        """
        Return the integer multiple of :code:`2 * f_nyquist` closest to
        :code:`DEFAULT_F_SAMPLING`, so that a time series sampled at
        :code:`2 * f_nyquist` can be resampled to it.
        """
        fmax = 2 * f_nyquist
        return int(fmax * np.round(cls.DEFAULT_F_SAMPLING / fmax))

    def _create_sky_samples(self):
        """Return Earth-fixed latitudes and longitudes, isotropic, from a Halton sequence."""
        u_lat, u_lon = qmc.Halton(2, seed=self._rng).random(self.nsky).T
        return {"lat": np.arcsin(2 * u_lat - 1), "lon": 2 * np.pi * u_lon}

    def _create_delays2inds_map(self):
        """Return the map from each cell, a tuple of rounded delays, to the indices of its sky locations."""
        if len(self.detector_names) == 1:
            return {(): np.arange(self.nsky)}
        keys, inverse = np.unique(
            self._discretize(self.delays).T, axis=0, return_inverse=True
        )
        order = np.argsort(inverse.ravel(), kind="stable")
        groups = np.split(order, np.cumsum(np.bincount(inverse.ravel()))[:-1])
        return {
            tuple(int(value) for value in key): group
            for key, group in zip(keys, groups)
        }

    def _discretize(self, delays):
        """Return delays (s) rounded to integer multiples of 1 / f_sampling."""
        return np.rint(delays * self.f_sampling).astype(int)

    def apply_tdet_prior(self, t_arrival_lnprob):
        """
        Add to the log probabilities of arrival time, in place, the prior of
        the delays between interferometers given the probabilities at the
        louder ones (arXiv:2404.02435 Eqs. 36-38), as cogwheel's
        :code:`SkyDictionary.apply_tdet_prior`. Interferometers are ranked by
        the largest value of their log probability.

        Parameters
        ----------
        t_arrival_lnprob: array_like
            Of shape (number of interferometers, number of times).
        """
        det_order = tuple(np.argsort(t_arrival_lnprob.max(axis=1)))[::-1]
        if len(det_order) > 1:
            prob_t0 = exp_normalize(t_arrival_lnprob[det_order[0]])
            t_arrival_lnprob[det_order[1]] += self._second_detector_lnprior(
                det_order, prob_t0
            )
        if len(det_order) > 2:
            prob_t1 = exp_normalize(t_arrival_lnprob[det_order[1]])
            t_arrival_lnprob[det_order[2]] += self._third_detector_lnprior(
                det_order, prob_t0, prob_t1
            )

    def _second_detector_lnprior(self, det_order, prob_t0):
        """Log prior of the arrival time at the second interferometer given that at the first (Eq. 37)."""
        prior_t1 = scipy.signal.convolve(
            prob_t0, self._delays_prior[det_order[:2]], "same"
        )
        return np.log(prior_t1 + 1e-10)

    def _third_detector_lnprior(self, det_order, prob_t0, prob_t1):
        """Log prior of the arrival time at the third interferometer given those at the first two (Eq. 38)."""
        prob_dt_01 = scipy.signal.correlate(prob_t1, prob_t0, "same")
        dt_01 = scipy.signal.correlation_lags(len(prob_t1), len(prob_t0), "same")
        min_delay, max_delay = self._delays_bounds[det_order[:2]]
        mask = (dt_01 >= min_delay) & (dt_01 <= max_delay)
        prior_dt02 = prob_dt_01[mask] @ self._delays_prior[det_order[:3]]
        prior_t2 = scipy.signal.convolve(prob_t0, prior_dt02, "same")
        return np.log(prior_t2 + 1e-10)


def n_effective(weights):
    """The effective sample size, :math:`(\\sum w)^2 / \\sum w^2`, as cogwheel's :code:`utils.n_effective`."""
    weights = np.asarray(weights)
    if weights.size == 0:
        return 0.0
    return np.sum(weights) ** 2 / np.sum(weights**2)


@dataclass
class MarginalizationInfo:
    """
    The result of one marginalization over the extrinsic parameters, as
    cogwheel's :code:`MarginalizationInfo` and :code:`MarginalizationInfoHM`:
    the kept (sample, phase) pairs, the inner products and the proposals
    they were drawn from, and the derived weights, effective sample size
    and marginalized likelihood.

    The estimate is arXiv:2404.02435 Eqs. 31-32 (multiple importance
    sampling with the balance heuristic):

    .. math::

        \\hat L = \\sum_{(q, o)} \\frac{\\exp(\\text{ln\\_numerators}_{qo})}
        {\\sum_j n_j \\prod_I P_{j,I}(\\tau_{qI})}

    with :math:`n_j` the number of samples drawn from proposal :math:`P_j`,
    counting the samples dropped by the physical mask or by the time prior.

    Attributes
    ----------
    qmc_sequence_id: int
        Index of the QMC sequence used.
    ln_numerators: array_like
        For each kept pair: log of the likelihood marginalized over
        distance, plus the log of the sky and time prior factor, minus the
        log of the number of orbital phases.
    q_inds, o_inds: array_like
        For each kept pair: index in the QMC sequence and of the orbital
        phase.
    sky_inds: array_like
        For each kept pair: index of the sky location in the sky dictionary.
    t_first_det: array_like
        For each kept pair: arrival time at the first interferometer (s,
        on the scale of the time grid).
    d_h, h_h: array_like
        For each kept pair: the real :math:`\\langle d|h\\rangle` and
        :math:`\\langle h|h\\rangle` of the network.
    tdet_inds: array_like
        Of shape (interferometer, kept pair): arrival-time indices.
    flip_psi: array_like
        Boolean, for each kept pair: whether :math:`\\langle d|h\\rangle < 0`.
    proposals_n_qmc: list
        Number of samples drawn from each proposal.
    proposals: list
        Each proposal, of shape (interferometer, time), rows summing to one.
    weights: array_like
        Normalized importance weights of the kept pairs.
    weights_q: array_like
        Normalized weights summed over the orbital phases of each sample,
        the nonzero ones.
    n_qmc: int
        Number of samples drawn, including those dropped.
    n_effective: float
        Effective sample size of :code:`weights_q`.
    lnl_marginalized: float
        Log of the likelihood marginalized over the extrinsic parameters;
        :code:`-inf` if no pair is kept.
    """

    qmc_sequence_id: int
    ln_numerators: np.ndarray
    q_inds: np.ndarray
    o_inds: np.ndarray
    sky_inds: np.ndarray
    t_first_det: np.ndarray
    d_h: np.ndarray
    h_h: np.ndarray
    tdet_inds: np.ndarray
    flip_psi: np.ndarray
    proposals_n_qmc: list
    proposals: list

    weights: np.ndarray = field(init=False)
    weights_q: np.ndarray = field(init=False)
    n_qmc: int = field(init=False)
    n_effective: float = field(init=False)
    lnl_marginalized: float = field(init=False)

    def __post_init__(self):
        """Set the derived attributes, as cogwheel's :code:`MarginalizationInfo.__post_init__`."""
        self.n_qmc = sum(self.proposals_n_qmc)
        if self.q_inds.size == 0:
            self.weights = np.array([])
            self.weights_q = np.array([])
            self.n_effective = 0.0
            self.lnl_marginalized = -np.inf
            return
        denominators = np.zeros(len(self.q_inds))
        for n_qmc, proposal in zip(self.proposals_n_qmc, self.proposals):
            denominators += n_qmc * np.prod(
                np.take_along_axis(proposal, self.tdet_inds, axis=1), axis=0
            )
        ln_weights = self.ln_numerators - np.log(denominators)
        self.weights = exp_normalize(ln_weights)
        weights_q = np.bincount(
            self.q_inds, weights=self.weights
        )  # the phases of each sample summed
        self.weights_q = weights_q[weights_q > 0]
        self.n_effective = n_effective(self.weights_q)
        self.lnl_marginalized = logsumexp(ln_weights)

    def update(self, other):
        """
        Add the pairs of :code:`other`, from a later chunk of the same QMC
        sequence, and recompute the derived attributes, as cogwheel's
        :code:`MarginalizationInfo.update`.
        """
        if self.qmc_sequence_id != other.qmc_sequence_id:
            raise ValueError("Cannot use different QMC sequences.")
        self.proposals_n_qmc = self.proposals_n_qmc + other.proposals_n_qmc
        self.proposals = self.proposals + other.proposals
        for attribute in (
            "ln_numerators",
            "q_inds",
            "o_inds",
            "sky_inds",
            "t_first_det",
            "d_h",
            "h_h",
            "tdet_inds",
            "flip_psi",
        ):
            setattr(
                self,
                attribute,
                np.concatenate(
                    [getattr(self, attribute), getattr(other, attribute)], axis=-1
                ),
            )
        self.__post_init__()


class CoherentScore:
    """
    The likelihood marginalized over the extrinsic parameters, from the
    matched-filter time series, following arXiv:2404.02435 Sec. II and
    cogwheel's :code:`CoherentScoreHM`.

    For each extrinsic sample (sky location, arrival times and polarization)
    the orbital phase is integrated with the trapezoid rule on
    :code:`nphi` points and the distance with a lookup table
    (arXiv:2404.02435 Eqs. 14-21).

    Parameters
    ----------
    sky_dict: SkyDictionary
        The sky locations grouped by arrival-time delays.
    m_arr: array_like
        The azimuthal index m of each harmonic, in the order of the time
        series.
    lnlike_marginalized_over_distance: callable
        :code:`f(d_inner_h, h_inner_h)`, the log of the likelihood
        marginalized over distance with its normalized prior, for real
        :math:`\\langle d|h\\rangle` and :math:`\\langle h|h\\rangle` of the
        waveform at the distance it was generated at; vectorized.
    nphi: int, optional
        Number of orbital phases of the trapezoid rule. Default is 128.
    dlnl_threshold: float, optional
        In nats. A (sample, phase) pair is dropped from the orbital-phase
        integral when its log likelihood maximized over distance,
        :math:`\\langle d|h\\rangle^2 / (2 \\langle h|h\\rangle)`, is more
        than this below the largest over all pairs (arXiv:2404.02435 Sec. A6).
        Default is 12, cogwheel's :code:`BaseCoherentScoreHM.DLNL_THRESHOLD`;
        :code:`numpy.inf` keeps every pair.
    beta_temperature: float or array_like, optional
        Inverse temperature of the incoherent arrival-time proposal, one
        value or one per interferometer (arXiv:2404.02435 Eq. 34, beta_d). Default
        is 0.5, as cogwheel.
    learning_rate: float, optional
        Weight of each new adapted proposal in the remembered arrival-time
        proposal (arXiv:2404.02435 Eq. A9, :math:`\\epsilon`); 0 turns the
        memory off. Default is 0.01, as cogwheel.
    log2n_qmc: int, optional
        Base-2 logarithm of the number of extrinsic samples drawn per
        iteration of the importance sampling (one chunk of a QMC sequence).
        Default is 11, as cogwheel.
    max_log2n_qmc: int, optional
        Base-2 logarithm of the length of each QMC sequence, the largest
        number of extrinsic samples. Default is 15, as cogwheel.
    n_qmc_sequences: int, optional
        Number of QMC sequences; each call uses the next one in turn, as
        cogwheel. Default is 128.
    seed: int, optional
        Seed of the QMC sequences. Default is 0.
    min_n_effective: float, optional
        The adaptive importance sampling adds chunks of the QMC sequence
        until the effective sample size reaches this. Default is 50, as
        cogwheel.
    ln_time_prior: callable, optional
        :code:`f(t_geocenter)`, the log of the normalized prior density
        (1/s) of the arrival time at the geocenter, vectorized, with
        :code:`t_geocenter` on the scale of the time grid passed to
        :code:`get_marginalization_info`. With it, each sample's numerator
        carries this density and samples where it is :code:`-inf` are
        dropped, so that :code:`lnl_marginalized` estimates the likelihood
        averaged over a normalized prior of every extrinsic parameter.
        Default is :code:`None`: cogwheel's convention, a time prior of 1
        per second over the time grid (arXiv:2404.02435 Eqs. 27-28), so that
        the estimate exceeds the normalized one by the log of the width of
        the time prior in seconds
        (research/extrinsic-marginalization/2026-11-11_normalization-audit.md).

    Notes
    -----
    The other factors of the estimate are normalized as in cogwheel: the
    sky prior factor :math:`\\Delta \\Pi(\\delta\\tau)` of each sample, with
    :math:`\\Delta` the cell width and :math:`\\Pi` the fraction of the sky
    dictionary's locations in the cell; the polarization drawn from its
    uniform prior on (0, pi); the average over :code:`nphi` orbital phases;
    and the likelihood marginalized over distance with its normalized prior
    (:code:`lnlike_marginalized_over_distance`).
    """

    def __init__(
        self,
        sky_dict,
        m_arr,
        lnlike_marginalized_over_distance,
        nphi=128,
        dlnl_threshold=12.0,
        beta_temperature=0.5,
        learning_rate=1e-2,
        log2n_qmc=11,
        max_log2n_qmc=15,
        n_qmc_sequences=128,
        seed=0,
        min_n_effective=50,
        ln_time_prior=None,
    ):
        if max_log2n_qmc < log2n_qmc:
            raise ValueError("max_log2n_qmc < log2n_qmc")
        self.sky_dict = sky_dict
        self.m_arr = np.asarray(m_arr)
        self.m_inds, self.mprime_inds = (
            np.array(indices)
            for indices in zip(
                *itertools.combinations_with_replacement(range(len(self.m_arr)), 2)
            )
        )
        self.lnlike_marginalized_over_distance = lnlike_marginalized_over_distance
        self.nphi = nphi
        self.dlnl_threshold = dlnl_threshold
        self.beta_temperature = np.asarray(beta_temperature)
        self.learning_rate = learning_rate
        self.min_n_effective = min_n_effective
        self.ln_time_prior = ln_time_prior
        self._t_arrival_prob = None
        self.log2n_qmc = log2n_qmc
        self.max_log2n_qmc = max_log2n_qmc
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._current_qmc_sequence_id = 0
        # The QMC sequences need the interferometers, so they are made only with a sky dictionary
        self._qmc_sequences = (
            []
            if sky_dict is None
            else [self._create_qmc_sequence() for _ in range(n_qmc_sequences)]
        )
        self._qmc_ind_chunks = self._create_qmc_ind_chunks()

    @property
    def nphi(self):
        """
        Number of orbital phases of the trapezoid rule. Setting it also sets
        :code:`phi_ref`, the phases, and the phase factors
        :code:`_dh_phasor` (m, phase), :math:`e^{-i m \\phi}`, and
        :code:`_hh_phasor` (pair, phase), :math:`e^{i (m - m') \\phi}`, for a
        waveform whose harmonic m is :math:`h_m(0) e^{i m \\phi}`.
        """
        return self._nphi

    @nphi.setter
    def nphi(self, nphi):
        self._nphi = nphi
        self.phi_ref = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
        self._dh_phasor = np.exp(-1j * np.outer(self.m_arr, self.phi_ref))
        self._hh_phasor = np.exp(
            1j
            * np.outer(
                self.m_arr[self.m_inds] - self.m_arr[self.mprime_inds], self.phi_ref
            )
        )

    def get_dh_hh_qo(self, fplus_fcross, dh_dmpq, hh_mppd):
        """
        Return the real inner products :math:`\\langle d|h\\rangle` and
        :math:`\\langle h|h\\rangle` of the network for each sample and orbital
        phase (arXiv:2404.02435 Eqs. 15-20), as cogwheel's
        :code:`CoherentScoreHM._get_dh_hh_qo` after the interpolation.

        Parameters
        ----------
        fplus_fcross: array_like
            Antenna patterns, of shape (sample, interferometer, polarization).
        dh_dmpq: array_like
            Complex :math:`\\langle d|h_{m,p}\\rangle` at each sample's arrival
            time, of shape (interferometer, harmonic, polarization, sample).
        hh_mppd: array_like
            Complex :math:`\\langle h_{m,p}|h_{m',p'}\\rangle`, of shape (pair,
            polarization, polarization, interferometer), off-diagonal pairs
            doubled.

        Returns
        -------
        dh_qo, hh_qo: array_like
            Real, of shape (sample, phase).
        """
        dh_qm = np.einsum("dmpq,qdp->qm", dh_dmpq, fplus_fcross)
        hh_qm = np.einsum("qdp,qdP,kpPd->qk", fplus_fcross, fplus_fcross, hh_mppd)
        return (dh_qm @ self._dh_phasor).real, (hh_qm @ self._hh_phasor).real

    def get_lnnumerators_important_flippsi(self, dh_qo, hh_qo, sky_prior):
        """
        Return the log of the numerators of the importance-sampling weights
        of the (sample, phase) pairs that are kept, as cogwheel's
        :code:`BaseCoherentScoreHM._get_lnnumerators_important_flippsi`: the
        likelihood marginalized over distance, times the sky prior factor of
        the sample, over :code:`nphi` (arXiv:2404.02435 Eqs. 21 and 32).

        Parameters
        ----------
        dh_qo, hh_qo: array_like
            Real, of shape (sample, phase), from :code:`get_dh_hh_qo`.
        sky_prior: array_like
            The sky prior factor of each sample (s), from
            :code:`SkyDictionary.get_sky_inds_and_prior`.

        Returns
        -------
        ln_numerators: array_like
            For each kept pair.
        important: tuple
            Sample and phase indices of the kept pairs: those whose
            :math:`\\langle d|h\\rangle^2 / (2 \\langle h|h\\rangle)`, the log
            likelihood maximized over distance, is within
            :code:`dlnl_threshold` of the largest.
        flip_psi: array_like
            Boolean, for each kept pair: whether :math:`\\langle d|h\\rangle < 0`,
            used when drawing samples, as in cogwheel.
        """
        flip_psi = np.signbit(dh_qo)
        max_over_distance_lnl = 0.5 * dh_qo**2 / hh_qo
        threshold = np.max(max_over_distance_lnl) - self.dlnl_threshold
        important = np.where(max_over_distance_lnl > threshold)
        ln_numerators = (
            self.lnlike_marginalized_over_distance(dh_qo[important], hh_qo[important])
            + np.log(sky_prior[important[0]])
            - np.log(self.nphi)
        )
        return ln_numerators, important, flip_psi[important]

    def incoherent_t_arrival_lnprob(self, dh_mptd, hh_mppd):
        """
        Return the tempered log likelihood of each arrival time at each
        interferometer, maximized over distance and orbital phase as if the
        harmonics and polarizations were orthogonal with independent phases
        (arXiv:2404.02435 Eqs. 34-35), as cogwheel's
        :code:`CoherentScoreHM._incoherent_t_arrival_lnprob`:
        :math:`\\beta_I / 2 \\, (\\sum_{m,p} |\\langle d_I|h_{m,p}\\rangle(t)|)^2
        / \\sum_{m,p} \\langle h_{m,p}|h_{m,p}\\rangle_I`.

        Parameters
        ----------
        dh_mptd: array_like
            Complex :math:`\\langle d|h_{m,p}\\rangle` on the grid of arrival
            times, of shape (harmonic, polarization, time, interferometer).
        hh_mppd: array_like
            Complex :math:`\\langle h_{m,p}|h_{m',p'}\\rangle`, of shape (pair,
            polarization, polarization, interferometer).

        Returns
        -------
        array_like
            Of shape (interferometer, time).
        """
        hh_mpdiagonal = hh_mppd[np.equal(self.m_inds, self.mprime_inds)][
            :, (0, 1), (0, 1)
        ].real  # mpd
        chi_squared = np.einsum("mptd->td", np.abs(dh_mptd)) ** 2 / np.einsum(
            "mpd->d", hh_mpdiagonal
        )
        return (self.beta_temperature / 2 * chi_squared).T

    def get_t_arrival_proposal(self, dh_mptd, hh_mppd):
        """
        Return the proposal of arrival times at each interferometer, as the
        first part of cogwheel's :code:`BaseCoherentScore.get_marginalization_info`
        (arXiv:2404.02435 Eqs. 34-38 and A9): the incoherent proposal, with
        the prior of the delays between interferometers given the louder
        ones, normalized; then, if a proposal is remembered from earlier
        calls, the equal-weight mixture of the two.

        Parameters
        ----------
        dh_mptd: array_like
            Complex :math:`\\langle d|h_{m,p}\\rangle` on the time grid of
            :code:`sky_dict` (see :code:`SkyDictionary.resample_timeseries`),
            of shape (harmonic, polarization, time, interferometer).
        hh_mppd: array_like
            Complex :math:`\\langle h_{m,p}|h_{m',p'}\\rangle`, of shape (pair,
            polarization, polarization, interferometer).

        Returns
        -------
        array_like
            Probabilities of shape (interferometer, time), each row summing to
            one.
        """
        t_arrival_lnprob = self.incoherent_t_arrival_lnprob(dh_mptd, hh_mppd)
        self.sky_dict.apply_tdet_prior(t_arrival_lnprob)
        t_arrival_prob = exp_normalize(t_arrival_lnprob)
        if self._t_arrival_prob is not None:
            if self._t_arrival_prob.shape == t_arrival_prob.shape:
                t_arrival_prob = 0.5 * (t_arrival_prob + self._t_arrival_prob)
            else:
                logger.warning(
                    "The length of the time series changed, resetting the arrival-time proposal."
                )
                self._t_arrival_prob = None
        return t_arrival_prob

    def remember_t_arrival_proposal(self, proposal):
        """
        Add an adapted arrival-time proposal to the remembered one, with
        weight :code:`learning_rate`, and normalize (arXiv:2404.02435 Eq.
        A9), as the end of cogwheel's
        :code:`BaseCoherentScore.get_marginalization_info`. The first
        proposal is remembered as it is. Does nothing if
        :code:`learning_rate` is 0. cogwheel calls it only when the effective
        sample size exceeds its minimum; that check is the caller's.

        Parameters
        ----------
        proposal: array_like
            Probabilities of shape (interferometer, time), each row summing to
            one: the last proposal of the adaptive importance sampling.
        """
        if not self.learning_rate:
            return
        if self._t_arrival_prob is None:
            self._t_arrival_prob = np.array(proposal, dtype=float)
        else:
            self._t_arrival_prob += self.learning_rate * proposal
            self._t_arrival_prob /= self._t_arrival_prob.sum(axis=1, keepdims=True)

    @property
    def _qmc_range_dic(self):
        """
        Ranges of the QMC sequence, as cogwheel's
        :code:`BaseCoherentScore._qmc_range_dic`: the cumulative probability of
        the arrival time at each interferometer, the arrival time within a
        cell of width :code:`1 / f_sampling` (s) and the polarization (rad).
        """
        dt = 1 / self.sky_dict.f_sampling
        return dict.fromkeys(self.sky_dict.detector_names, (0, 1)) | {
            "t_fine": (-dt / 2, dt / 2),
            "psi": (0, np.pi),
        }

    def _create_qmc_sequence(self):
        """
        Return a scrambled Halton sequence of :code:`2 ** max_log2n_qmc`
        points over :code:`_qmc_range_dic`, as cogwheel's
        :code:`BaseCoherentScore._create_qmc_sequence`: a dictionary with
        :code:`u_tdet` (interferometer, sample), :code:`t_fine` and
        :code:`psi` (sample), and :code:`rot_psi` (sample, 2, 2), the matrices
        that turn the antenna patterns at polarization zero into those at
        :code:`psi`.
        """
        sequence_values = qmc.scale(
            qmc.Halton(len(self._qmc_range_dic), seed=self._rng).random(
                2**self.max_log2n_qmc
            )
            % 1,  # 1 -> 0
            *zip(*self._qmc_range_dic.values()),
        ).T
        n_det = len(self.sky_dict.detector_names)
        qmc_sequence = dict(
            zip(list(self._qmc_range_dic)[n_det:], sequence_values[n_det:]),
            u_tdet=sequence_values[:n_det],
        )
        sintwopsi = np.sin(2 * qmc_sequence["psi"])
        costwopsi = np.cos(2 * qmc_sequence["psi"])
        qmc_sequence["rot_psi"] = np.moveaxis(
            [[costwopsi, sintwopsi], [-sintwopsi, costwopsi]], -1, 0
        ).astype(np.float32)
        return qmc_sequence

    def _create_qmc_ind_chunks(self):
        """The indices of a QMC sequence, split into chunks of :code:`2 ** log2n_qmc`."""
        qmc_inds = np.arange(2**self.max_log2n_qmc)
        breaks = np.arange(2**self.log2n_qmc, 2**self.max_log2n_qmc, 2**self.log2n_qmc)
        return np.split(qmc_inds, breaks)

    @property
    def n_qmc_sequences(self):
        """Number of QMC sequences to cycle through."""
        return len(self._qmc_sequences)

    @property
    def _qmc_sequence(self):
        return self._qmc_sequences[self._current_qmc_sequence_id]

    def switch_qmc_sequence(self, qmc_sequence_id=None):
        """
        Use the QMC sequence :code:`qmc_sequence_id`, or the next one in turn
        if it is :code:`None`, as cogwheel's
        :code:`BaseCoherentScore._switch_qmc_sequence`, called at the start
        of each marginalization.
        """
        if qmc_sequence_id is None:
            qmc_sequence_id = (self._current_qmc_sequence_id + 1) % self.n_qmc_sequences
        self._current_qmc_sequence_id = qmc_sequence_id

    def get_tdet_inds(self, t_arrival_prob, q_inds):
        """
        Return the indices of the arrival times at each interferometer for
        the QMC samples :code:`q_inds`, by inverse transform sampling of
        :code:`t_arrival_prob`, as cogwheel's
        :code:`BaseCoherentScore._get_tdet_inds` (arXiv:2404.02435 Sec. II D 2).

        Parameters
        ----------
        t_arrival_prob: array_like
            Probabilities of shape (interferometer, time), each row summing to
            one.
        q_inds: array_like
            Indices into the current QMC sequence.

        Returns
        -------
        array_like
            Integers, of shape (interferometer, sample).
        """
        cumprob = np.cumsum(t_arrival_prob, axis=1)
        np.testing.assert_allclose(cumprob[:, -1], 1)
        return np.array(
            [
                np.searchsorted(cumulative, u)
                for cumulative, u in zip(
                    cumprob, self._qmc_sequence["u_tdet"][:, q_inds]
                )
            ]
        )

    def draw_extrinsic_samples(self, t_arrival_prob, times, i_chunk):
        """
        Draw extrinsic samples from the arrival-time proposal, with chunk
        :code:`i_chunk` of the current QMC sequence, as the first part of
        cogwheel's :code:`CoherentScoreHM._get_marginalization_info_chunk`
        (arXiv:2404.02435 Eqs. 23-28): arrival times at each interferometer
        on the time grid, the sky location of their rounded delays, and the
        arrival time at the first interferometer within its cell.
        Samples whose delays no sky location has are dropped.

        Parameters
        ----------
        t_arrival_prob: array_like
            Probabilities of shape (interferometer, time), each row summing to
            one, on :code:`times`.
        times: array_like
            The time grid, of spacing :code:`1 / sky_dict.f_sampling` (s).
        i_chunk: int
            Index into the chunks of the QMC sequence.

        Returns
        -------
        dict
            :code:`n_qmc`, the number of samples drawn; :code:`physical_mask`,
            Boolean of length :code:`n_qmc`; and for each kept sample:
            :code:`q_inds`, its index in the QMC sequence; :code:`tdet_inds`
            (interferometer, sample), its arrival-time indices;
            :code:`sky_inds`; :code:`sky_prior`, the factor
            :math:`\\Delta \\Pi(\\delta\\tau)` (s); :code:`t_first_det`, the
            arrival time at the first interferometer (s, on the scale of
            :code:`times`); and :code:`psi`, the polarization (rad).
        """
        q_inds = self._qmc_ind_chunks[i_chunk]
        n_qmc = len(q_inds)
        tdet_inds = self.get_tdet_inds(t_arrival_prob, q_inds)
        sky_inds, sky_prior, physical_mask = self.sky_dict.get_sky_inds_and_prior(
            tdet_inds[1:] - tdet_inds[0]
        )
        q_inds = q_inds[physical_mask]
        tdet_inds = tdet_inds[:, physical_mask]
        t_first_det = times[tdet_inds[0]] + self._qmc_sequence["t_fine"][q_inds]
        return dict(
            n_qmc=n_qmc,
            physical_mask=physical_mask,
            q_inds=q_inds,
            tdet_inds=tdet_inds,
            sky_inds=sky_inds,
            sky_prior=sky_prior,
            t_first_det=t_first_det,
            psi=self._qmc_sequence["psi"][q_inds],
        )

    def get_fplus_fcross(self, sky_inds, q_inds):
        """
        Return the antenna patterns of the samples, of shape (sample,
        interferometer, polarization), at their sky locations and
        polarizations, as cogwheel's :code:`BaseCoherentScore._get_fplus_fcross`.
        """
        fplus_fcross_0 = self.sky_dict.fplus_fcross_0[sky_inds]
        rot_psi = self._qmc_sequence["rot_psi"][q_inds]
        return (rot_psi[:, np.newaxis] @ fplus_fcross_0[..., np.newaxis])[..., 0]

    @staticmethod
    def interp_locally(times, timeseries, new_times, spline_degree=3):
        """
        Interpolate :code:`timeseries` along its last axis at
        :code:`new_times` with a spline through the points that span them,
        as cogwheel's :code:`BaseCoherentScore._interp_locally`; with too few
        points, an interpolating polynomial.
        """
        t_rng = new_times.min(), new_times.max()
        i_min, i_max = np.clip(
            np.searchsorted(times, t_rng) + (-1, 1), 0, len(times) - 1
        )
        if i_max - i_min > spline_degree:
            return make_interp_spline(
                times[i_min:i_max],
                timeseries[..., i_min:i_max],
                k=spline_degree,
                check_finite=False,
                axis=-1,
            )(new_times)
        if i_min == i_max:
            return timeseries[..., (i_min,) * len(new_times)]
        return krogh_interpolate(
            times[i_min:i_max], timeseries[..., i_min:i_max], new_times, axis=-1
        )

    def get_marginalization_info_chunk(
        self, dh_mptd, hh_mppd, times, t_arrival_prob, i_chunk
    ):
        """
        Draw the samples of chunk :code:`i_chunk` of the current QMC
        sequence from :code:`t_arrival_prob` and return their
        :code:`MarginalizationInfo`, as cogwheel's
        :code:`CoherentScoreHM._get_marginalization_info_chunk`. With
        :code:`ln_time_prior`, samples whose arrival time at the geocenter
        has zero prior density are dropped and the density enters the
        numerators of the others.

        Parameters
        ----------
        dh_mptd: array_like
            Complex :math:`\\langle d|h_{m,p}\\rangle` on :code:`times`, of
            shape (harmonic, polarization, time, interferometer).
        hh_mppd: array_like
            Complex :math:`\\langle h_{m,p}|h_{m',p'}\\rangle`, of shape (pair,
            polarization, polarization, interferometer).
        times: array_like
            The time grid, of spacing :code:`1 / sky_dict.f_sampling` (s).
        t_arrival_prob: array_like
            The proposal, of shape (interferometer, time), rows summing to one.
        i_chunk: int
            Index into the chunks of the QMC sequence.

        Returns
        -------
        MarginalizationInfo
        """
        draws = self.draw_extrinsic_samples(t_arrival_prob, times, i_chunk)
        q_inds, tdet_inds, t_first_det = (
            draws["q_inds"],
            draws["tdet_inds"],
            draws["t_first_det"],
        )
        sky_inds, sky_prior = (
            np.asarray(draws["sky_inds"], dtype=int),
            draws["sky_prior"],
        )
        if self.ln_time_prior is not None and len(q_inds) > 0:
            # The time prior: arXiv:2404.02435 Eq. 27 with pi(t_geocenter) normalized, not 1 per second
            ln_time_prior = np.asarray(
                self.ln_time_prior(
                    t_first_det - self.sky_dict.geocenter_delay_first_det[sky_inds]
                ),
                dtype=float,
            )
            inside = ln_time_prior > -np.inf
            q_inds, tdet_inds, t_first_det, sky_inds = (
                q_inds[inside],
                tdet_inds[:, inside],
                t_first_det[inside],
                sky_inds[inside],
            )
            sky_prior = sky_prior[inside] * np.exp(ln_time_prior[inside])
        if len(q_inds) == 0:
            return MarginalizationInfo(
                qmc_sequence_id=self._current_qmc_sequence_id,
                ln_numerators=np.array([]),
                q_inds=np.array([], int),
                o_inds=np.array([], int),
                sky_inds=np.array([], int),
                t_first_det=np.array([]),
                d_h=np.array([]),
                h_h=np.array([]),
                tdet_inds=tdet_inds,
                flip_psi=np.array([], bool),
                proposals_n_qmc=[draws["n_qmc"]],
                proposals=[t_arrival_prob],
            )

        fplus_fcross = self.get_fplus_fcross(sky_inds, q_inds)
        t_det = np.vstack(
            (t_first_det, t_first_det + self.sky_dict.delays[:, sky_inds])
        )
        dh_dmpq = np.array(
            [
                self.interp_locally(times, dh_mptd[..., i_det], t_det[i_det])
                for i_det in range(len(self.sky_dict.detector_names))
            ]
        )
        dh_qo, hh_qo = self.get_dh_hh_qo(fplus_fcross, dh_dmpq, hh_mppd)
        ln_numerators, important, flip_psi = self.get_lnnumerators_important_flippsi(
            dh_qo, hh_qo, sky_prior
        )
        return MarginalizationInfo(
            qmc_sequence_id=self._current_qmc_sequence_id,
            ln_numerators=ln_numerators,
            q_inds=q_inds[important[0]],
            o_inds=important[1],
            sky_inds=sky_inds[important[0]],
            t_first_det=t_first_det[important[0]],
            d_h=dh_qo[important],
            h_h=hh_qo[important],
            tdet_inds=tdet_inds[:, important[0]],
            flip_psi=flip_psi,
            proposals_n_qmc=[draws["n_qmc"]],
            proposals=[t_arrival_prob],
        )

    def kde_t_arrival_prob(self, marginalization_info, times):
        """
        Return a proposal of arrival times, of shape (interferometer, time),
        rows summing to one, from a kernel density estimate of the weighted
        samples so far: their histogram convolved with a Cauchy kernel of
        scale Silverman's factor times their standard deviation, at least
        one cell, as cogwheel's :code:`BaseCoherentScore._kde_t_arrival_prob`
        (arXiv:2404.02435 Sec. II D 2).
        """
        info = marginalization_info
        delays = self.sky_dict.delays[:, info.sky_inds]
        t_det = np.vstack((info.t_first_det, info.t_first_det + delays))
        silverman_factor = (3 / 4 * info.n_effective) ** -0.2
        dt = times[1] - times[0]
        bins = np.concatenate([[-np.inf], times[:-1] + dt / 2, [np.inf]])
        prob = []
        for t_samples in t_det:
            hist = np.histogram(t_samples, weights=info.weights, bins=bins)[0]
            average = np.average(t_samples, weights=info.weights)
            std = np.sqrt(np.average((t_samples - average) ** 2, weights=info.weights))
            scale = max(dt, silverman_factor * std)
            kernel = cauchy.pdf(times, loc=times[(len(times) - 1) // 2], scale=scale)
            prob.append(scipy.signal.convolve(hist, kernel, mode="same"))
        prob = np.array(prob)
        return prob / prob.sum(axis=1, keepdims=True)

    def adapt_marginalization_info(
        self,
        dh_mptd,
        hh_mppd,
        times,
        t_arrival_prob,
        lnl_marginalized_threshold=-np.inf,
    ):
        """
        Run the adaptive importance sampling on the next QMC sequence, as
        cogwheel's :code:`BaseCoherentScore._get_marginalization_info`
        (arXiv:2404.02435 Eqs. 30-33, Sec. A8): after each chunk, while the
        effective sample size is below :code:`min_n_effective`, mix half the
        kernel density estimate of the samples with half the last proposal
        and draw the next chunk from it. It stops at the last chunk, when no
        sample is kept, or, once the effective sample size exceeds 2, when
        :code:`lnl_marginalized` is below :code:`lnl_marginalized_threshold`.

        Returns
        -------
        MarginalizationInfo
        """
        self.switch_qmc_sequence()
        i_chunk = 0
        info = self.get_marginalization_info_chunk(
            dh_mptd, hh_mppd, times, t_arrival_prob, i_chunk
        )
        while info.n_effective < self.min_n_effective:
            i_chunk += 1
            if i_chunk == len(self._qmc_ind_chunks):
                logger.debug("Maximum QMC resolution reached.")
                break
            if info.n_effective == 0:
                break
            if (
                info.n_effective > 2
                and info.lnl_marginalized < lnl_marginalized_threshold
            ):
                break
            t_arrival_prob = 0.5 * (
                self.kde_t_arrival_prob(info, times) + t_arrival_prob
            )
            info.update(
                self.get_marginalization_info_chunk(
                    dh_mptd, hh_mppd, times, t_arrival_prob, i_chunk
                )
            )
        return info

    def get_marginalization_info(
        self, dh_mptd, hh_mppd, times, lnl_marginalized_threshold=-np.inf
    ):
        """
        Return the likelihood marginalized over the extrinsic parameters,
        with its samples, as cogwheel's
        :code:`ProposingCoherentScore.get_marginalization_info`: resample the
        time series to the sky dictionary's sampling frequency, build the
        arrival-time proposal (:code:`get_t_arrival_proposal`), run the
        adaptive importance sampling (:code:`adapt_marginalization_info`)
        and, if the effective sample size exceeds :code:`min_n_effective`,
        remember the last proposal (:code:`remember_t_arrival_proposal`).

        Parameters
        ----------
        dh_mptd: array_like
            Complex :math:`\\langle d|h_{m,p}\\rangle` on :code:`times`, of
            shape (harmonic, polarization, time, interferometer).
        hh_mppd: array_like
            Complex :math:`\\langle h_{m,p}|h_{m',p'}\\rangle`, of shape (pair,
            polarization, polarization, interferometer).
        times: array_like
            The time grid of :code:`dh_mptd` (s), evenly spaced.
        lnl_marginalized_threshold: float, optional
            Stop adapting when the estimate falls below this, see
            :code:`adapt_marginalization_info`. Default is :code:`-inf`.

        Returns
        -------
        MarginalizationInfo
        """
        dh_mptd, times = self.sky_dict.resample_timeseries(dh_mptd, times, axis=-2)
        t_arrival_prob = self.get_t_arrival_proposal(dh_mptd, hh_mppd)
        info = self.adapt_marginalization_info(
            dh_mptd, hh_mppd, times, t_arrival_prob, lnl_marginalized_threshold
        )
        if self.learning_rate and info.n_effective > self.min_n_effective:
            self.remember_t_arrival_proposal(info.proposals[-1])
        return info

    def optimize_beta_temperature(self, dh_mptd, hh_mppd, times, beta_rng=(0.1, 1)):
        """
        Set :code:`beta_temperature`, one value per interferometer, to
        minimize the number of samples per effective sample, as cogwheel's
        :code:`ProposingCoherentScore.optimize_beta_temperature`: two passes
        over the interferometers, each trying 100 values spaced
        geometrically over :code:`beta_rng`, with :code:`min_n_effective`
        set to zero, and taking the minimum of the cost smoothed by a
        Savitzky-Golay filter. Each trial runs
        :code:`get_marginalization_info`, so it also updates the remembered
        proposal, as in cogwheel.
        """
        number = len(self.sky_dict.detector_names)
        beta_temperature = np.array(
            np.broadcast_to(self.beta_temperature, number), dtype=float
        )
        betas = np.geomspace(*beta_rng, 100)
        saved = self.beta_temperature, self.min_n_effective
        try:
            self.beta_temperature, self.min_n_effective = beta_temperature, 0
            for _ in range(2):
                for i_det in range(number):
                    cost = []
                    for beta in betas:
                        beta_temperature[i_det] = beta
                        info = self.get_marginalization_info(dh_mptd, hh_mppd, times)
                        cost.append(info.n_qmc / (info.n_effective + 1e-3))
                    cost_smooth = scipy.signal.savgol_filter(
                        cost, len(cost) // 10, 1, mode="nearest"
                    )
                    beta_temperature[i_det] = betas[np.argmin(cost_smooth)]
        finally:
            self.beta_temperature, self.min_n_effective = saved
        self.beta_temperature = beta_temperature

    def draw_posterior_samples(self, marginalization_info, num, rng, flip_psi=True):
        """
        Draw extrinsic samples from the conditional posterior that a
        marginalization estimated, as cogwheel's
        :code:`CoherentScoreHM.gen_samples_from_marg_info` without the
        distance: each sample is a kept (sample, phase) pair chosen with
        probability equal to its normalized importance weight.

        Parameters
        ----------
        marginalization_info: MarginalizationInfo
            From :code:`get_marginalization_info`.
        num: int
            Number of samples.
        rng: numpy.random.Generator
            The random generator.
        flip_psi: bool, optional
            Whether a pair with :math:`\\langle d|h\\rangle < 0` is returned
            with its polarization increased by pi/2 (modulo pi) and the sign
            of :math:`\\langle d|h\\rangle` reversed, as cogwheel
            (arXiv:2404.02435 Sec. A7). The antenna patterns change sign under
            that shift, so the pair's waveform changes sign. Default is
            :code:`True`, as cogwheel.

        Returns
        -------
        dict
            Arrays of length :code:`num`: :code:`lat` and :code:`lon`, the
            Earth-fixed latitude and longitude (rad); :code:`psi` (rad);
            :code:`phase`, the orbital phase of the pair (rad);
            :code:`t_geocenter`, the geocentric arrival time on the scale of
            the time grid (s); :code:`d_h` and :code:`h_h`, the real inner
            products of the pair at the distance the waveform was generated
            at, :code:`d_h` after the flip. All are NaN if no pair is kept.
        """
        info = marginalization_info
        keys = ["lat", "lon", "psi", "phase", "t_geocenter", "d_h", "h_h"]
        if info.q_inds.size == 0:
            return {key: np.full(num, np.nan) for key in keys}
        chosen = rng.choice(len(info.q_inds), size=num, p=info.weights)
        q_inds, sky_inds = info.q_inds[chosen], info.sky_inds[chosen]
        psi = self._qmc_sequences[info.qmc_sequence_id]["psi"][q_inds]
        d_h = info.d_h[chosen].astype(float)
        if flip_psi:
            flipped = info.flip_psi[chosen]
            psi = np.where(flipped, (psi + np.pi / 2) % np.pi, psi)
            d_h = np.where(flipped, -d_h, d_h)
        return dict(
            lat=self.sky_dict.sky_samples["lat"][sky_inds],
            lon=self.sky_dict.sky_samples["lon"][sky_inds],
            psi=psi,
            phase=self.phi_ref[info.o_inds[chosen]],
            t_geocenter=info.t_first_det[chosen]
            - self.sky_dict.geocenter_delay_first_det[sky_inds],
            d_h=d_h,
            h_h=info.h_h[chosen].astype(float),
        )
