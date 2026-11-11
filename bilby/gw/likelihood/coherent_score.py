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

import numpy as np
import scipy.signal
from scipy.stats import qmc

from ...core.utils import logger, speed_of_light
from ..geometry import get_polarization_tensor
from ..time import greenwich_mean_sidereal_time


def get_antenna_patterns_at_zero_polarization(interferometers, latitude, longitude, time):
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
        get_polarization_tensor(np.ravel(right_ascension), np.ravel(latitude), time, 0.0, mode)
        for mode in ["plus", "cross"]]
    antenna_patterns = np.array([
        [np.einsum("ij,ij...->...", interferometer.geometry.detector_tensor, tensor)
         for tensor in polarization_tensors]
        for interferometer in interferometers])
    return np.moveaxis(antenna_patterns, -1, 0).reshape(np.shape(latitude) + antenna_patterns.shape[:2])


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
    direction = np.array([np.cos(latitude) * np.cos(longitude),
                          np.cos(latitude) * np.sin(longitude),
                          np.sin(latitude)])
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
    return np.stack([cos_two_psi * plus + sin_two_psi * cross,
                     -sin_two_psi * plus + cos_two_psi * cross], axis=-1)


def exp_normalize(log_probability, axis=-1):
    """
    Return probabilities normalized to sum to one along :code:`axis`, from
    unnormalized log probabilities, as cogwheel's :code:`utils.exp_normalize`.
    """
    probability = np.exp(log_probability - np.max(log_probability, axis=axis, keepdims=True))
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
    DEFAULT_F_SAMPLING = 2 ** 13

    def __init__(self, interferometers, *, f_sampling=DEFAULT_F_SAMPLING, nsky=10 ** 6, seed=0):
        self.detector_names = tuple(interferometer.name for interferometer in interferometers)
        self.nsky = nsky
        self.f_sampling = f_sampling
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        self.sky_samples = self._create_sky_samples()
        # In Earth-fixed coordinates the antenna patterns do not depend on time; any GPS time gives the same values
        self.fplus_fcross_0 = get_antenna_patterns_at_zero_polarization(
            interferometers, self.sky_samples["lat"], self.sky_samples["lon"], 0.0).astype(np.float32)
        geocenter_delays = get_geocenter_delays(interferometers, self.sky_samples["lat"], self.sky_samples["lon"])
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
            for order in itertools.permutations(range(len(self.detector_names)), number):
                pairwise_delays = self._discretize(geocenter_delays[order[1:], ] - geocenter_delays[order[0]])
                bins = [np.arange(*bounds) + .5 for bounds in zip(
                    pairwise_delays.min(axis=1) - 1, pairwise_delays.max(axis=1) + 1)]
                self._delays_prior[order] = np.histogramdd(
                    pairwise_delays.T, weights=weights, bins=bins)[0].copy("F")
                if number == 2:
                    self._delays_bounds[order] = (pairwise_delays.min(), pairwise_delays.max())

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
        window = np.sqrt(scipy.signal.windows.tukey(shape[axis], .1)).reshape(shape)
        endpoint = np.mean(timeseries.take([0, -1], axis), axis, keepdims=True)
        timeseries = (timeseries - endpoint) * window + endpoint

        ratio = np.round(self.f_sampling * (times[1] - times[0]), decimals=10)
        if ratio != 1:
            timeseries, times = scipy.signal.resample(timeseries, int(len(times) * ratio), times, axis=axis)
            if not np.isclose(1 / self.f_sampling, times[1] - times[0]):
                raise ValueError("The spacing of `times` is incommensurate with `f_sampling`.")
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
            sky_inds = np.fromiter(itertools.islice(self._ind_generators[()], number), int, number)
            return sky_inds, np.full(number, self._sky_prior), np.full(number, True)

        physical_mask = np.all((delays.T >= self._min_delay) & (delays.T <= self._max_delay), axis=1)
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
        keys, inverse = np.unique(self._discretize(self.delays).T, axis=0, return_inverse=True)
        order = np.argsort(inverse.ravel(), kind="stable")
        groups = np.split(order, np.cumsum(np.bincount(inverse.ravel()))[:-1])
        return {tuple(int(value) for value in key): group for key, group in zip(keys, groups)}

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
            t_arrival_lnprob[det_order[1]] += self._second_detector_lnprior(det_order, prob_t0)
        if len(det_order) > 2:
            prob_t1 = exp_normalize(t_arrival_lnprob[det_order[1]])
            t_arrival_lnprob[det_order[2]] += self._third_detector_lnprior(det_order, prob_t0, prob_t1)

    def _second_detector_lnprior(self, det_order, prob_t0):
        """Log prior of the arrival time at the second interferometer given that at the first (Eq. 37)."""
        prior_t1 = scipy.signal.convolve(prob_t0, self._delays_prior[det_order[:2]], "same")
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
    """

    def __init__(self, sky_dict, m_arr, lnlike_marginalized_over_distance, nphi=128, dlnl_threshold=12.,
                 beta_temperature=0.5, learning_rate=1e-2, log2n_qmc=11, max_log2n_qmc=15, n_qmc_sequences=128,
                 seed=0):
        if max_log2n_qmc < log2n_qmc:
            raise ValueError("max_log2n_qmc < log2n_qmc")
        self.sky_dict = sky_dict
        self.m_arr = np.asarray(m_arr)
        self.m_inds, self.mprime_inds = (
            np.array(indices) for indices in zip(*itertools.combinations_with_replacement(range(len(self.m_arr)), 2)))
        self.lnlike_marginalized_over_distance = lnlike_marginalized_over_distance
        self.nphi = nphi
        self.dlnl_threshold = dlnl_threshold
        self.beta_temperature = np.asarray(beta_temperature)
        self.learning_rate = learning_rate
        self._t_arrival_prob = None
        self.log2n_qmc = log2n_qmc
        self.max_log2n_qmc = max_log2n_qmc
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._current_qmc_sequence_id = 0
        # The QMC sequences need the interferometers, so they are made only with a sky dictionary
        self._qmc_sequences = (
            [] if sky_dict is None else [self._create_qmc_sequence() for _ in range(n_qmc_sequences)])
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
        self._hh_phasor = np.exp(1j * np.outer(self.m_arr[self.m_inds] - self.m_arr[self.mprime_inds], self.phi_ref))

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
        max_over_distance_lnl = 0.5 * dh_qo ** 2 / hh_qo
        threshold = np.max(max_over_distance_lnl) - self.dlnl_threshold
        important = np.where(max_over_distance_lnl > threshold)
        ln_numerators = (self.lnlike_marginalized_over_distance(dh_qo[important], hh_qo[important])
                         + np.log(sky_prior[important[0]]) - np.log(self.nphi))
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
        hh_mpdiagonal = hh_mppd[np.equal(self.m_inds, self.mprime_inds)][:, (0, 1), (0, 1)].real  # mpd
        chi_squared = (np.einsum("mptd->td", np.abs(dh_mptd)) ** 2
                       / np.einsum("mpd->d", hh_mpdiagonal))
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
                logger.warning("The length of the time series changed, resetting the arrival-time proposal.")
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
        return dict.fromkeys(self.sky_dict.detector_names, (0, 1)) | {"t_fine": (-dt / 2, dt / 2), "psi": (0, np.pi)}

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
            qmc.Halton(len(self._qmc_range_dic), seed=self._rng).random(2 ** self.max_log2n_qmc) % 1,  # 1 -> 0
            *zip(*self._qmc_range_dic.values())).T
        n_det = len(self.sky_dict.detector_names)
        qmc_sequence = dict(zip(list(self._qmc_range_dic)[n_det:], sequence_values[n_det:]),
                            u_tdet=sequence_values[:n_det])
        sintwopsi = np.sin(2 * qmc_sequence["psi"])
        costwopsi = np.cos(2 * qmc_sequence["psi"])
        qmc_sequence["rot_psi"] = np.moveaxis(
            [[costwopsi, sintwopsi], [-sintwopsi, costwopsi]], -1, 0).astype(np.float32)
        return qmc_sequence

    def _create_qmc_ind_chunks(self):
        """The indices of a QMC sequence, split into chunks of :code:`2 ** log2n_qmc`."""
        qmc_inds = np.arange(2 ** self.max_log2n_qmc)
        breaks = np.arange(2 ** self.log2n_qmc, 2 ** self.max_log2n_qmc, 2 ** self.log2n_qmc)
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
        return np.array([np.searchsorted(cumulative, u)
                         for cumulative, u in zip(cumprob, self._qmc_sequence["u_tdet"][:, q_inds])])

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
        sky_inds, sky_prior, physical_mask = self.sky_dict.get_sky_inds_and_prior(tdet_inds[1:] - tdet_inds[0])
        q_inds = q_inds[physical_mask]
        tdet_inds = tdet_inds[:, physical_mask]
        t_first_det = times[tdet_inds[0]] + self._qmc_sequence["t_fine"][q_inds]
        return dict(n_qmc=n_qmc, physical_mask=physical_mask, q_inds=q_inds, tdet_inds=tdet_inds,
                    sky_inds=sky_inds, sky_prior=sky_prior, t_first_det=t_first_det,
                    psi=self._qmc_sequence["psi"][q_inds])
