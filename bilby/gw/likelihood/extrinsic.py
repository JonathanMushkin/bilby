import numpy as np

from .relative_modes import ModeByModeRelativeBinningGravitationalWaveTransient
from ...core.prior import Uniform
from ...core.utils import logger, radius_of_earth, speed_of_light


class ExtrinsicMarginalizedRelativeBinningGravitationalWaveTransient(
        ModeByModeRelativeBinningGravitationalWaveTransient):
    """A gravitational-wave transient likelihood object which marginalizes
    over the extrinsic parameters (sky location, arrival time,
    polarization, orbital phase and distance) with the algorithm of
    Roulet, Mushkin et al., arXiv:2404.02435, and its implementation in
    cogwheel, on top of the mode-by-mode relative binning of
    :code:`ModeByModeRelativeBinningGravitationalWaveTransient`.

    For each interferometer, harmonic m and polarization, the inner product
    of the data with the waveform is computed on a grid of arrival times at
    the interferometer, the matched-filter time series :math:`z_{mpd}(t)`
    of arXiv:2404.02435 Eq. 11, from summary data that carry the time shift
    exactly. With :code:`linear_free`, each waveform is first time-aligned
    to the fiducial waveform, as cogwheel's :code:`BaseLinearFree`
    (arXiv:2404.02435 Eqs. A6-A8), and the grid of arrival times moves by
    the same amount.

    Parameters
    ----------
    interferometers: list, bilby.gw.detector.InterferometerList
        A list of `bilby.detector.Interferometer` instances - contains the
        detector data and power spectral densities
    waveform_generator: `bilby.waveform_generator.WaveformGenerator`
        An object which computes the frequency-domain strain of the signal,
        given some set of parameters; it must use
        :func:`bilby.gw.source.lal_binary_black_hole_relative_binning_by_mode`
        or another source model with the same output.
    priors: dict
        Must contain a :code:`Uniform` prior on :code:`geocent_time`, whose
        range sets the grid of arrival times.
    fiducial_parameters: dict, optional
        See :code:`ModeByModeRelativeBinningGravitationalWaveTransient`.
    chi: float, optional
        See :code:`ModeByModeRelativeBinningGravitationalWaveTransient`.
    epsilon: float, optional
        See :code:`ModeByModeRelativeBinningGravitationalWaveTransient`.
    spline_degree: int, optional
        See :code:`ModeByModeRelativeBinningGravitationalWaveTransient`.
    safe_fiducial_parameters: bool, optional
        See :code:`ModeByModeRelativeBinningGravitationalWaveTransient`.
    linear_free: bool, optional
        Whether to time-align each waveform to the fiducial waveform before
        the relative binning, see :code:`get_linear_free_time_shift`.
        Default is :code:`True`, as cogwheel.
    dlnl_threshold: float, optional
        In nats. In the integral over orbital phase, a (sample, phase) pair
        is dropped when its log likelihood maximized over distance is more
        than this below the largest (arXiv:2404.02435 Sec. A6), see
        :code:`CoherentScore`. Default is 12, as cogwheel.
    time_padding: float, optional
        In seconds, the time added at each end of the grid of arrival times
        beyond the range of the :code:`geocent_time` prior widened by the
        light travel time across the radius of the Earth. Default is
        :code:`None`, a tenth of that widened range. The resampling of the
        time series tapers 5% of the grid at each end, so the grid serves
        every arrival time of the prior only while the linear-free time
        shift is at most :code:`time_shift_margin`; see
        :code:`check_time_shift`.

    Notes
    -----
    Calibration marginalization and calibration models are not supported.
    """

    def __init__(self, interferometers,
                 waveform_generator,
                 priors,
                 fiducial_parameters=None,
                 chi=1,
                 epsilon=0.5,
                 spline_degree=3,
                 safe_fiducial_parameters=True,
                 linear_free=True,
                 dlnl_threshold=12.,
                 time_padding=None):
        self.dlnl_threshold = dlnl_threshold
        self.times, self.time_padding = self._get_time_grid(
            interferometers, priors, waveform_generator, time_padding)
        self._largest_time_shift_warned = None
        self.time_series_summary_data = dict()
        self.linear_free = linear_free
        self._linear_free_reference = None
        self._linear_free_weights = None
        super().__init__(
            interferometers=interferometers,
            waveform_generator=waveform_generator,
            fiducial_parameters=fiducial_parameters,
            priors=priors,
            chi=chi,
            epsilon=epsilon,
            spline_degree=spline_degree,
            safe_fiducial_parameters=safe_fiducial_parameters)

    @staticmethod
    def _get_time_grid(interferometers, priors, waveform_generator, time_padding=None):
        """
        Return the grid of arrival times at the interferometers, relative to
        the start time of the data, and its padding: the range of the
        :code:`geocent_time` prior, widened on each side by the light travel
        time across the radius of the Earth and then by :code:`time_padding`
        (a tenth of the widened range if :code:`None`), with spacing half the
        inverse of the largest frequency, as cogwheel
        (:code:`MarginalizedExtrinsicLikelihood`, which uses a fixed range of
        0.14 s). The times are kept relative to the start time because a
        GPS time in float64 resolves only about 2e-7 s.

        Parameters
        ----------
        interferometers: bilby.gw.detector.InterferometerList
            Gives the start time of the data.
        priors: dict
            The priors, with a :code:`Uniform` prior on :code:`geocent_time`.
        waveform_generator: bilby.gw.waveform_generator.WaveformGenerator
            Gives the largest frequency.
        time_padding: float, optional
            In seconds; see the class.

        Returns
        -------
        times: array_like
            The arrival times minus the start time of the data (s).
        time_padding: float
            In seconds.
        """
        if priors is None or not isinstance(priors.get("geocent_time"), Uniform):
            raise ValueError("A Uniform prior on geocent_time is required.")
        start_time = interferometers.start_time
        minimum = priors["geocent_time"].minimum - start_time - radius_of_earth / speed_of_light
        maximum = priors["geocent_time"].maximum - start_time + radius_of_earth / speed_of_light
        padding = (maximum - minimum) / 10 if time_padding is None else float(time_padding)
        spacing = 1 / (2 * waveform_generator.frequency_array[-1])
        return np.arange(minimum - padding, maximum + padding, spacing), padding

    # The fraction of the time grid tapered at each end when the time series are resampled
    # (SkyDictionary.resample_timeseries, a Tukey window with parameter 0.1)
    _TAPERED_FRACTION = 0.05

    @property
    def time_shift_margin(self):
        """
        In seconds, the largest linear-free time shift for which the
        untapered part of the time grid holds every arrival time of the
        :code:`geocent_time` prior: the padding minus the length tapered at
        each end.
        """
        duration = self.times[-1] - self.times[0]
        return self.time_padding - self._TAPERED_FRACTION * duration

    def required_time_padding(self, time_shift):
        """
        In seconds, the smallest :code:`time_padding` whose
        :code:`time_shift_margin` is at least :code:`abs(time_shift)`.
        """
        unpadded = self.times[-1] - self.times[0] - 2 * self.time_padding
        return (abs(time_shift) + self._TAPERED_FRACTION * unpadded) / (1 - 2 * self._TAPERED_FRACTION)

    def check_time_shift(self, time_shift):
        """
        Log a warning if :code:`abs(time_shift)`, a linear-free time shift
        in seconds, exceeds :code:`time_shift_margin`, and more than any
        shift warned about before. Then part of the :code:`geocent_time`
        prior falls in a tapered end of the time grid or off it, and its
        likelihood is distorted or zero. The warning names the
        :code:`time_padding` that would cover the shift.

        Returns
        -------
        bool
            Whether the shift is within the margin.
        """
        if abs(time_shift) <= self.time_shift_margin:
            return True
        if self._largest_time_shift_warned is None or abs(time_shift) > self._largest_time_shift_warned:
            self._largest_time_shift_warned = abs(time_shift)
            logger.warning(
                f"The linear-free time shift {time_shift:.4g} s exceeds the margin of the time grid, "
                f"{self.time_shift_margin:.4g} s: part of the geocent_time prior is not covered. Set "
                f"time_padding to at least {self.required_time_padding(time_shift):.4g} s to cover it.")
        return False

    def compute_summary_data(self):
        r"""
        Compute the summary data of
        :code:`ModeByModeRelativeBinningGravitationalWaveTransient` and, for
        each interferometer, harmonic m and arrival time :math:`t_s + t_k`,
        with :math:`t_k` in :code:`times`, the weights

        .. math::

            D_{m,k,j} = \frac{4}{T} \sum_f S_j(f) \frac{h_{0,m}^*(f) d(f)}{P(f)}
            e^{2 \pi i f (t_k - t_0)} \Big/ \left(h_{0,m}^*(f_j)
            e^{-2 \pi i f_j t_0}\right)

        of :math:`\langle h|d \rangle` at bin edge j, with :math:`h_{0,m}`
        the fiducial harmonic after the fade of the parent class,
        :math:`t_s + t_0` its arrival time at the interferometer and
        :math:`t_s` the start time of the data. The time shift :math:`e^{2 \pi i f
        (t_k - t_0)}` is part of the integrand, so only the ratio of the
        waveform to the fiducial waveform, both at arrival time :math:`t_s`,
        is interpolated between bin edges.
        """
        names = [interferometer.name for interferometer in self.interferometers]
        number_of_harmonics = len(self.per_detector_fiducial_mode_waveforms[names[0]])
        super().compute_summary_data()

        # The fade of the parent class (relative_modes.py, compute_summary_data), repeated here because the faded
        # fiducial waveforms are not stored
        inverse_psds = dict()
        for interferometer in self.interferometers:
            inverse_psds[interferometer.name] = np.zeros(len(interferometer.frequency_array))
            inverse_psds[interferometer.name][interferometer.frequency_mask] = (
                1 / interferometer.power_spectral_density_array[interferometer.frequency_mask])
        network_inverse_psd = sum(inverse_psds.values())
        frequency_array = self.interferometers[0].frequency_array
        parameters = self.fiducial_waveform_parameters

        for interferometer in self.interferometers:
            name = interferometer.name
            time_shift = np.exp(-2j * np.pi * frequency_array * (
                parameters["geocent_time"] - interferometer.strain_data.start_time))
            h0 = self.per_detector_fiducial_mode_waveforms[name] / time_shift
            h0_bins = h0[:, self.bin_inds]
            for harmonic in range(number_of_harmonics):
                self._fade_ringdown(h0[harmonic], h0_bins[harmonic], frequency_array, network_inverse_psd)
            h0 *= time_shift
            h0_bins *= time_shift[self.bin_inds]

            # t_0: the arrival time of h0 at the interferometer minus the start time, as in get_detector_response
            arrival_time = parameters["geocent_time"] - interferometer.strain_data.start_time + (
                interferometer.time_delay_from_geocenter(parameters["ra"], parameters["dec"], parameters["geocent_time"]))
            integrand = h0.conj() * interferometer.frequency_domain_strain * inverse_psds[name]
            weights = np.array([
                self._get_summary_weights(
                    integrand * np.exp(2j * np.pi * frequency_array * (time - arrival_time)), interferometer.duration)
                for time in self.times])  # (time, harmonic, bin edge)
            denominator = h0_bins.conj() * np.exp(-2j * np.pi * self.bin_freqs * arrival_time)
            self.time_series_summary_data[name] = np.moveaxis(weights, 0, 1) / denominator[:, np.newaxis]

        # The reference and weights of the linear-free time convention: the plus polarization of the m = 2 harmonic of the
        # fiducial waveform, faded, and the square root of the summary weights of its squared amplitude over the
        # network noise, as cogwheel's BaseLinearFree._set_summary
        reference = np.copy(self.fiducial_polarizations["plus"][self._harmonic_index(2)])
        reference_bins = reference[self.bin_inds]
        self._fade_ringdown(reference, reference_bins, frequency_array, network_inverse_psd)
        self._linear_free_reference = reference_bins
        self._linear_free_weights = np.sqrt(np.clip(self._get_summary_weights(
            np.abs(reference) ** 2 * network_inverse_psd, self.interferometers[0].duration).real, 0, None))

    def _harmonic_index(self, m):
        """
        Return the index of harmonic m in the output of the source model,
        whose harmonics are in the order of first appearance in its
        :code:`mode_array` (:func:`lal_binary_black_hole_relative_binning_by_mode`).
        """
        mode_array = self.waveform_generator.waveform_arguments.get(
            "mode_array", [[2, 2], [2, 1], [3, 3], [3, 2], [4, 4]])
        return list(dict.fromkeys(int(mode[1]) for mode in mode_array)).index(m)

    def get_linear_free_time_shift(self, waveform_polarizations):
        """
        Return the time shift that aligns the waveform with the fiducial
        waveform, as cogwheel's
        :code:`BaseLinearFree._get_linearfree_hplus_hcross_dt`: minus the
        slope, divided by 2 pi, of a weighted straight-line fit to the
        unwrapped phase of the ratio of the plus polarizations of the m = 2
        harmonics at the bin edges. The fit weights are those of the
        fiducial waveform times the square root of the ratio's modulus.

        Parameters
        ----------
        waveform_polarizations: dict
            Plus and cross polarizations at the bin edges, each of shape
            (number of harmonics, number of bin edges), with no time shift.

        Returns
        -------
        time_shift: float
            In seconds; the waveform times :math:`e^{2 \\pi i f \\delta}` is
            aligned with the fiducial waveform, :math:`\\delta` the time shift.
        """
        ratio = waveform_polarizations["plus"][self._harmonic_index(2)] / self._linear_free_reference
        phase = np.unwrap(np.angle(ratio))
        fit = np.polynomial.Polynomial.fit(
            self.bin_freqs, phase, deg=1, w=self._linear_free_weights * np.sqrt(np.abs(ratio)))
        return -fit.deriv()(0) / (2 * np.pi)

    def get_time_series_inner_products(self, waveform_polarizations):
        """
        Return the inner products of the data with each harmonic and
        polarization of the waveform, on the grid of arrival times, and of
        the waveform with itself, for each interferometer, without the
        antenna patterns. With :code:`linear_free`, the waveform is first
        multiplied by :math:`e^{2 \\pi i f \\delta}`, :math:`\\delta` the time
        shift of :code:`get_linear_free_time_shift`, so that the time series
        at grid time :code:`times[k]` is that of the waveform arriving at
        :code:`times[k] - time_shift`; otherwise the time shift is zero.

        Parameters
        ----------
        waveform_polarizations: dict
            Plus and cross polarizations at the bin edges, each of shape
            (number of harmonics, number of bin edges), with no time shift.

        Returns
        -------
        d_inner_h: array_like
            Complex, of shape (harmonic, polarization, time, interferometer):
            :math:`\\langle d|h_{m,p}\\rangle` with the waveform arriving at
            the start time plus :code:`times[k] - time_shift`;
            :math:`z_{mpd}(t)` of arXiv:2404.02435 Eq. 11.
        h_inner_h: array_like
            Complex, of shape (pair of harmonics, polarization, polarization,
            interferometer): :math:`\\langle h_{m,p}|h_{m',p'}\\rangle` for
            the pairs m <= m' of :code:`harmonic_pairs`, off-diagonal pairs
            doubled; :math:`c_{mm'pp'd}` of arXiv:2404.02435 Eq. 13.
        time_shift: float
            In seconds; see above.
        """
        waveform = np.array([waveform_polarizations["plus"], waveform_polarizations["cross"]])  # (p, m, j)
        time_shift = self.get_linear_free_time_shift(waveform_polarizations) if self.linear_free else 0.0
        waveform = waveform * np.exp(2j * np.pi * self.bin_freqs * time_shift)
        m_inds, mprime_inds = self.harmonic_pairs
        d_inner_h, h_inner_h = list(), list()
        for interferometer in self.interferometers:
            d_inner_h.append(np.einsum(
                "mtj,pmj->mpt", self.time_series_summary_data[interferometer.name], waveform.conj()))
            h_inner_h_weights = self.summary_data[interferometer.name][1]
            h_inner_h.append(np.einsum(
                "kj,pkj,qkj->kpq", h_inner_h_weights, waveform[:, m_inds], waveform[:, mprime_inds].conj()))
        return np.stack(d_inner_h, axis=-1), np.stack(h_inner_h, axis=-1), time_shift
