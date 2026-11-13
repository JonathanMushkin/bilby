import unittest
from copy import deepcopy

import bilby
import numpy as np
from parameterized import parameterized

from scipy.special import logsumexp

from bilby.gw.likelihood.coherent_score import (
    CoherentScore,
    MarginalizationInfo,
    SkyDictionary,
    exp_normalize,
    get_antenna_patterns_at_zero_polarization,
    get_geocenter_delays,
    rotate_antenna_patterns,
)
from bilby.gw.time import greenwich_mean_sidereal_time

GEOCENT_TIME = 1126259642.413
MINIMUM_FREQUENCY = 20
SAMPLING_FREQUENCY = 2048
DURATION = 4
WAVEFORM_ARGUMENTS = dict(
    waveform_approximant="IMRPhenomXPHM",
    reference_frequency=MINIMUM_FREQUENCY,
    minimum_frequency=MINIMUM_FREQUENCY,
    mode_array=[[2, 2], [2, 1], [3, 3], [3, 2], [4, 4]],
    PhenomXPrecVersion=102,
    PhenomXHMThresholdMband=0,
    PhenomXPHMThresholdMband=0,
)
INJECTION = dict(
    chirp_mass=30.0, mass_ratio=0.5, a_1=0.5, a_2=0.3, tilt_1=1.0, tilt_2=2.0, phi_12=1.0, phi_jl=0.5,
    theta_jn=0.8, psi=0.5, phase=1.3, ra=1.3, dec=-1.2, geocent_time=GEOCENT_TIME, luminosity_distance=1000.0)


def make_waveform_generator(source_model):
    return bilby.gw.WaveformGenerator(
        duration=DURATION, sampling_frequency=SAMPLING_FREQUENCY, frequency_domain_source_model=source_model,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=deepcopy(WAVEFORM_ARGUMENTS))


def make_likelihood(interferometer_names=("H1", "L1"), epsilon=0.05, linear_free=True):
    """Inject :code:`INJECTION` into zero noise and return the extrinsic-marginalized likelihood with the
    injection as fiducial parameters, and the interferometers."""
    interferometers = bilby.gw.detector.InterferometerList(list(interferometer_names))
    interferometers.set_strain_data_from_zero_noise(
        sampling_frequency=SAMPLING_FREQUENCY, duration=DURATION, start_time=GEOCENT_TIME - DURATION + 2)
    for interferometer in interferometers:
        interferometer.minimum_frequency = MINIMUM_FREQUENCY
    interferometers.inject_signal(
        waveform_generator=make_waveform_generator(bilby.gw.source.lal_binary_black_hole),
        parameters=INJECTION, raise_error=False)
    priors = bilby.gw.prior.BBHPriorDict()
    priors["geocent_time"] = bilby.core.prior.Uniform(GEOCENT_TIME - 0.1, GEOCENT_TIME + 0.1)
    likelihood = bilby.gw.likelihood.ExtrinsicMarginalizedRelativeBinningGravitationalWaveTransient(
        interferometers=interferometers,
        waveform_generator=make_waveform_generator(bilby.gw.source.lal_binary_black_hole_relative_binning_by_mode),
        priors=priors, fiducial_parameters=dict(INJECTION), epsilon=epsilon, linear_free=linear_free)
    return likelihood, interferometers


class TestSkyGeometry(unittest.TestCase):
    """The vectorized antenna patterns and delays against the scalar
    methods of :code:`Interferometer`."""

    def setUp(self):
        self.interferometers = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])
        rng = np.random.default_rng(0)
        self.latitude = np.arcsin(rng.uniform(-1, 1, 100))
        self.longitude = rng.uniform(0, 2 * np.pi, 100)
        self.psi = rng.uniform(0, np.pi, 100)
        self.right_ascension = self.longitude + greenwich_mean_sidereal_time(GEOCENT_TIME)

    def scalar_antenna_patterns(self, psi):
        return np.array([
            [[interferometer.antenna_response(ra, dec, GEOCENT_TIME, angle, mode) for mode in ["plus", "cross"]]
             for interferometer in self.interferometers]
            for ra, dec, angle in zip(self.right_ascension, self.latitude, psi)])

    def test_antenna_patterns_at_zero_polarization(self):
        antenna_patterns = get_antenna_patterns_at_zero_polarization(
            self.interferometers, self.latitude, self.longitude, GEOCENT_TIME)
        self.assertEqual(antenna_patterns.shape, (100, 3, 2))
        np.testing.assert_allclose(antenna_patterns, self.scalar_antenna_patterns(np.zeros(100)), atol=1e-12)

    def test_rotated_antenna_patterns(self):
        antenna_patterns = rotate_antenna_patterns(get_antenna_patterns_at_zero_polarization(
            self.interferometers, self.latitude, self.longitude, GEOCENT_TIME), self.psi)
        np.testing.assert_allclose(antenna_patterns, self.scalar_antenna_patterns(self.psi), atol=1e-12)

    def test_geocenter_delays(self):
        delays = get_geocenter_delays(self.interferometers, self.latitude, self.longitude)
        self.assertEqual(delays.shape, (3, 100))
        expected = np.array([
            [interferometer.time_delay_from_geocenter(ra, dec, GEOCENT_TIME)
             for ra, dec in zip(self.right_ascension, self.latitude)]
            for interferometer in self.interferometers])
        np.testing.assert_allclose(delays, expected, atol=1e-14)

    def test_shapes(self):
        latitude = self.latitude.reshape(10, 10)
        longitude = self.longitude.reshape(10, 10)
        self.assertEqual(get_antenna_patterns_at_zero_polarization(
            self.interferometers, latitude, longitude, GEOCENT_TIME).shape, (10, 10, 3, 2))
        self.assertEqual(get_antenna_patterns_at_zero_polarization(
            self.interferometers, 0.3, 1.2, GEOCENT_TIME).shape, (3, 2))
        self.assertEqual(get_geocenter_delays(self.interferometers, latitude, longitude).shape, (3, 10, 10))


NETWORKS = [("H1",), ("H1", "L1"), ("H1", "L1", "V1")]


class TestSkyDictionary(unittest.TestCase):
    """The sky dictionary for one, two and three interferometers."""

    @staticmethod
    def make(names):
        return SkyDictionary(bilby.gw.detector.InterferometerList(list(names)), nsky=10 ** 5)

    @parameterized.expand(NETWORKS)
    def test_cell_priors_sum_to_one(self, *names):
        sky_dict = self.make(names)
        self.assertAlmostEqual(sky_dict._sky_prior.sum() * sky_dict.f_sampling, 1)
        self.assertEqual(sum(len(inds) for inds in sky_dict.delays2inds_map.values()), sky_dict.nsky)

    @parameterized.expand(NETWORKS[1:])
    def test_points_lie_in_their_cells(self, *names):
        # every sky location's delays, in units of 1 / f_sampling, are within half a unit of its cell
        sky_dict = self.make(names)
        for key, inds in sky_dict.delays2inds_map.items():
            np.testing.assert_array_less(
                np.abs(sky_dict.delays[:, inds] * sky_dict.f_sampling - np.array(key)[:, np.newaxis]), 0.5 + 1e-9)

    @parameterized.expand(NETWORKS[1:])
    def test_sky_inds_and_prior(self, *names):
        sky_dict = self.make(names)
        keys = list(sky_dict.delays2inds_map)[:50]
        unphysical = tuple(int(value) + 10 ** 4 for value in keys[0])
        delays = np.array(keys + [unphysical]).T
        sky_inds, sky_prior, physical_mask = sky_dict.get_sky_inds_and_prior(delays)
        np.testing.assert_array_equal(physical_mask, [True] * len(keys) + [False])
        for key, ind, prior in zip(keys, sky_inds, sky_prior):
            self.assertIn(ind, sky_dict.delays2inds_map[key])
            self.assertAlmostEqual(prior, len(sky_dict.delays2inds_map[key]) / sky_dict.nsky / sky_dict.f_sampling)
        sky_dict.set_generators()
        np.testing.assert_array_equal(sky_dict.get_sky_inds_and_prior(delays)[0], sky_inds)

    def test_one_interferometer(self):
        sky_dict = self.make(("H1",))
        sky_inds, sky_prior, physical_mask = sky_dict.get_sky_inds_and_prior(np.zeros((0, 7), dtype=int))
        self.assertTrue(np.all(physical_mask))
        np.testing.assert_allclose(sky_prior, 1 / sky_dict.f_sampling)
        np.testing.assert_array_equal(sky_inds, np.arange(7))

    @parameterized.expand(NETWORKS)
    def test_apply_tdet_prior(self, *names):
        # a narrow peak at each interferometer, at arrival times consistent with one sky location, louder at the
        # first; the time axis, in units of 1 / f_sampling, is longer than any delay between interferometers, as
        # cogwheel's proposal requires
        sky_dict = self.make(names)
        times = np.arange(-600, 601)
        sky_ind = 12345
        arrivals = np.concatenate([[0], sky_dict._discretize(sky_dict.delays[:, sky_ind])])
        lnprob = -0.5 * ((times - arrivals[:, np.newaxis]) / 3.0) ** 2 + 10.0 * np.arange(len(names), 0, -1)[:, np.newaxis]
        modified = lnprob.copy()
        sky_dict.apply_tdet_prior(modified)
        self.assertTrue(np.all(np.isfinite(modified)))
        np.testing.assert_array_equal(modified[0], lnprob[0])  # the loudest interferometer is unchanged
        for row in range(1, len(names)):
            # the prior keeps the consistent arrival time near the peak
            self.assertLess(abs(times[np.argmax(modified[row])] - arrivals[row]), 3)

    def test_resample_timeseries(self):
        sky_dict = self.make(("H1",))
        times = np.arange(0, 0.1, 1 / 2048)
        signal = np.cos(2 * np.pi * 100 * times)
        resampled, new_times = sky_dict.resample_timeseries(signal, times)
        self.assertAlmostEqual(new_times[1] - new_times[0], 1 / sky_dict.f_sampling)
        middle = (new_times > 0.02) & (new_times < 0.08)
        np.testing.assert_allclose(resampled[middle], np.cos(2 * np.pi * 100 * new_times[middle]), atol=1e-2)

    def test_exp_normalize(self):
        probability = exp_normalize(np.array([[0.0, np.log(3.0)], [1000.0, 1000.0]]))
        np.testing.assert_allclose(probability, [[0.25, 0.75], [0.5, 0.5]])


def exact_distance_marginalization(reference_distance, distances=np.linspace(100.0, 5000.0, 40001)):
    """The log likelihood marginalized over distance with a prior proportional to distance squared on 100 to
    5000 Mpc, by the trapezoid rule on 40001 points, for <d|h> and <h|h> of a waveform at reference_distance."""
    prior = distances ** 2 / np.trapezoid(distances ** 2, distances)

    def lnlike(d_inner_h, h_inner_h):
        ratio = reference_distance / distances
        log_l = (np.asarray(d_inner_h)[..., np.newaxis] * ratio
                 - np.asarray(h_inner_h)[..., np.newaxis] * ratio ** 2 / 2)
        peak = log_l.max(axis=-1, keepdims=True)
        return np.log(np.trapezoid(np.exp(log_l - peak) * prior, distances)) + peak[..., 0]
    return lnlike


class TestPhaseAndDistanceMarginalization(unittest.TestCase):
    """Block 5: the inner products per sample and orbital phase, and the
    likelihood marginalized over phase and distance."""

    def setUp(self):
        self.m_arr = np.array([2, 1, 3, 4])
        rng = np.random.default_rng(1)
        self.number_of_samples, self.number_of_ifos, self.number_of_frequencies = 5, 2, 50
        self.waveform = (rng.normal(size=(4, 2, self.number_of_frequencies))
                         + 1j * rng.normal(size=(4, 2, self.number_of_frequencies))) * np.array(
            [1, 0.3, 0.2, 0.1])[:, np.newaxis, np.newaxis]  # (m, p, f), harmonics at phase 0
        self.data = rng.normal(size=(self.number_of_ifos, self.number_of_frequencies)) + 1j * rng.normal(
            size=(self.number_of_ifos, self.number_of_frequencies))
        self.fplus_fcross = rng.normal(size=(self.number_of_samples, self.number_of_ifos, 2))
        self.score = CoherentScore(None, self.m_arr, exact_distance_marginalization(1000.0), nphi=128)

    def inner_products(self):
        # with <a|b> = sum_f a conj(b): <d|h_mp> per interferometer and sample, and <h_mp|h_m'p'> per pair
        dh_dmpq = np.einsum("df,mpf->dmp", self.data, self.waveform.conj())[..., np.newaxis].repeat(
            self.number_of_samples, axis=-1)
        m_inds, mprime_inds = self.score.m_inds, self.score.mprime_inds
        hh = np.einsum("kpf,kPf->kpP", self.waveform[m_inds], self.waveform[mprime_inds].conj())
        hh[m_inds != mprime_inds] *= 2
        hh_mppd = hh[..., np.newaxis].repeat(self.number_of_ifos, axis=-1)
        return dh_dmpq, hh_mppd

    def test_against_the_waveform_at_each_phase(self):
        # harmonic m at phase phi is h_m(0) exp(i m phi); the phase factors must reproduce the network <d|h>, <h|h>
        dh_qo, hh_qo = self.score.get_dh_hh_qo(self.fplus_fcross, *self.inner_products())
        for i_phase in [0, 5, 37, 100]:
            phase = self.score.phi_ref[i_phase]
            harmonics = self.waveform * np.exp(1j * self.m_arr * phase)[:, np.newaxis, np.newaxis]
            for sample in range(self.number_of_samples):
                strain = np.einsum("dp,mpf->df", self.fplus_fcross[sample], harmonics)
                self.assertAlmostEqual(dh_qo[sample, i_phase], np.sum(self.data * strain.conj()).real, places=9)
                self.assertAlmostEqual(hh_qo[sample, i_phase], np.sum(np.abs(strain) ** 2), places=9)

    def test_phase_and_distance_integral(self):
        # with every pair kept, the numerators summed over phase are the phase average, on 128 points, of the
        # distance-marginalized likelihood; it is compared with the same average on 4096 points
        dh_qo, hh_qo = self.score.get_dh_hh_qo(self.fplus_fcross, *self.inner_products())
        self.score.dlnl_threshold = np.inf
        sky_prior = np.full(self.number_of_samples, 1.0)
        ln_numerators, important, flip_psi = self.score.get_lnnumerators_important_flippsi(dh_qo, hh_qo, sky_prior)
        self.assertEqual(len(ln_numerators), self.number_of_samples * 128)
        fine = CoherentScore(None, self.m_arr, self.score.lnlike_marginalized_over_distance, nphi=4096)
        dh_fine, hh_fine = fine.get_dh_hh_qo(self.fplus_fcross, *self.inner_products())
        expected = logsumexp(fine.lnlike_marginalized_over_distance(dh_fine, hh_fine), axis=1) - np.log(4096)
        for sample in range(self.number_of_samples):
            self.assertAlmostEqual(logsumexp(ln_numerators[important[0] == sample]), expected[sample], delta=1e-6)
        np.testing.assert_array_equal(flip_psi, dh_qo[important] < 0)

    def test_threshold(self):
        dh_qo = np.array([[30.0, -30.0, 10.0, 0.0]])
        hh_qo = np.full((1, 4), 30.0)
        ln_numerators, important, flip_psi = self.score.get_lnnumerators_important_flippsi(
            dh_qo, hh_qo, np.array([2.0]))
        # 0.5 dh^2 / hh: 15, 15, 1.67, 0; kept within 12 nats of 15
        np.testing.assert_array_equal(important[1], [0, 1])
        np.testing.assert_array_equal(flip_psi, [False, True])
        expected = self.score.lnlike_marginalized_over_distance(dh_qo[0, :2], hh_qo[0, :2]) + np.log(2.0) - np.log(128)
        np.testing.assert_allclose(ln_numerators, expected)

    def test_threshold_argument(self):
        # a threshold passed at construction replaces the default of 12 nats
        dh_qo = np.array([[30.0, -30.0, 10.0, 0.0]])
        hh_qo = np.full((1, 4), 30.0)
        self.assertEqual(self.score.dlnl_threshold, 12.0)
        wide = CoherentScore(None, self.m_arr, self.score.lnlike_marginalized_over_distance, dlnl_threshold=14.0)
        _, important, _ = wide.get_lnnumerators_important_flippsi(dh_qo, hh_qo, np.array([1.0]))
        # 0.5 dh^2 / hh: 15, 15, 1.67, 0; kept within 14 nats of 15
        np.testing.assert_array_equal(important[1], [0, 1, 2])



class TestArrivalTimeProposal(unittest.TestCase):
    """The proposal of arrival times at the interferometers, for one, two
    and three interferometers."""
    m_arr = [2, 1, 3, 4]
    number_of_times = 1200  # longer than the largest delay range in cells (H1-V1, 435)

    @staticmethod
    def make_score(names, **kwargs):
        sky_dict = SkyDictionary(bilby.gw.detector.InterferometerList(list(names)), nsky=10 ** 5)
        return CoherentScore(sky_dict, TestArrivalTimeProposal.m_arr, None, n_qmc_sequences=2, **kwargs)

    def time_series(self, number_of_ifos, seed=2):
        # noise in every harmonic, polarization and interferometer, plus a peak at a different time in each
        rng = np.random.default_rng(seed)
        shape = (len(self.m_arr), 2, self.number_of_times, number_of_ifos)
        dh_mptd = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        for i_ifo in range(number_of_ifos):
            dh_mptd[0, 0, 600 + 20 * i_ifo, i_ifo] += 15.0 / (i_ifo + 1)
        n_pairs = len(self.m_arr) * (len(self.m_arr) + 1) // 2
        hh_mppd = np.zeros((n_pairs, 2, 2, number_of_ifos), dtype=complex)
        hh_mppd[:, 0, 0] = hh_mppd[:, 1, 1] = 1.0 + np.arange(number_of_ifos)
        return dh_mptd, hh_mppd

    def test_incoherent_single_harmonic(self):
        # with one harmonic and polarization, the incoherent log likelihood is that maximized over distance and
        # phase, |<d|h>|^2 / (2 <h|h>), times beta_temperature
        score = self.make_score(("H1",), beta_temperature=0.7)
        dh_mptd = np.zeros((4, 2, 5, 1), dtype=complex)
        dh_mptd[1, 0, :, 0] = np.array([1, 2j, -3, 4 + 4j, 0])
        hh_mppd = np.zeros((10, 2, 2, 1), dtype=complex)
        diagonal = np.flatnonzero(score.m_inds == score.mprime_inds)
        hh_mppd[diagonal[1], 0, 0, 0] = 2.5
        expected = 0.7 * np.abs(dh_mptd[1, 0, :, 0]) ** 2 / (2 * 2.5)
        np.testing.assert_allclose(score.incoherent_t_arrival_lnprob(dh_mptd, hh_mppd)[0], expected)

    @parameterized.expand(NETWORKS)
    def test_proposal_is_normalized(self, *names):
        score = self.make_score(names)
        proposal = score.get_t_arrival_proposal(*self.time_series(len(names)))
        self.assertEqual(proposal.shape, (len(names), self.number_of_times))
        np.testing.assert_allclose(proposal.sum(axis=1), 1)
        self.assertTrue(np.all(proposal >= 0))

    @parameterized.expand(NETWORKS)
    def test_proposal_without_memory(self, *names):
        # the incoherent log likelihood plus the delay prior of SkyDictionary.apply_tdet_prior, normalized
        score = self.make_score(names)
        dh_mptd, hh_mppd = self.time_series(len(names))
        expected = score.incoherent_t_arrival_lnprob(dh_mptd, hh_mppd)
        score.sky_dict.apply_tdet_prior(expected)
        np.testing.assert_allclose(score.get_t_arrival_proposal(dh_mptd, hh_mppd), exp_normalize(expected))
        if len(names) == 1:  # no delay prior with one interferometer
            np.testing.assert_allclose(
                score.get_t_arrival_proposal(dh_mptd, hh_mppd),
                exp_normalize(score.incoherent_t_arrival_lnprob(dh_mptd, hh_mppd)))

    @parameterized.expand(NETWORKS)
    def test_memory(self, *names):
        score = self.make_score(names, learning_rate=0.1)
        dh_mptd, hh_mppd = self.time_series(len(names))
        first = score.get_t_arrival_proposal(dh_mptd, hh_mppd)
        score.remember_t_arrival_proposal(first)
        np.testing.assert_array_equal(score._t_arrival_prob, first)
        self.assertIsNot(score._t_arrival_prob, first)
        # with a proposal remembered, the new one is the equal-weight mixture
        other_dh = self.time_series(len(names), seed=3)[0]
        fresh = self.make_score(names).get_t_arrival_proposal(other_dh, hh_mppd)
        np.testing.assert_allclose(score.get_t_arrival_proposal(other_dh, hh_mppd), 0.5 * (fresh + first))
        # an adapted proposal enters the memory with weight learning_rate, then the rows are normalized
        adapted = exp_normalize(np.random.default_rng(4).normal(size=first.shape))
        score.remember_t_arrival_proposal(adapted)
        expected = (first + 0.1 * adapted) / (first + 0.1 * adapted).sum(axis=1, keepdims=True)
        np.testing.assert_allclose(score._t_arrival_prob, expected)

    def test_memory_off_and_reset(self):
        score = self.make_score(("H1", "L1"), learning_rate=0)
        dh_mptd, hh_mppd = self.time_series(2)
        score.remember_t_arrival_proposal(score.get_t_arrival_proposal(dh_mptd, hh_mppd))
        self.assertIsNone(score._t_arrival_prob)
        # a remembered proposal of another length is dropped
        score._t_arrival_prob = np.full((2, 10), 0.1)
        proposal = score.get_t_arrival_proposal(dh_mptd, hh_mppd)
        self.assertIsNone(score._t_arrival_prob)
        np.testing.assert_allclose(proposal, self.make_score(("H1", "L1")).get_t_arrival_proposal(dh_mptd, hh_mppd))


class TestDrawExtrinsicSamples(unittest.TestCase):
    """The QMC sequences and the extrinsic samples drawn from an
    arrival-time proposal, for one, two and three interferometers."""
    number_of_times = 1200  # longer than the largest delay range in cells (H1-V1, 435)

    @staticmethod
    def make_score(names, **kwargs):
        sky_dict = SkyDictionary(bilby.gw.detector.InterferometerList(list(names)), nsky=10 ** 5)
        return CoherentScore(sky_dict, [2, 1, 3, 4], None, n_qmc_sequences=3, **kwargs)

    def proposal(self, number_of_ifos, centers=None):
        # a Gaussian in each interferometer, 30 cells wide, centred 40 cells apart unless given
        cells = np.arange(self.number_of_times)
        centers = 500 + 40 * np.arange(number_of_ifos) if centers is None else np.asarray(centers)
        return exp_normalize(-0.5 * ((cells - centers[:, np.newaxis]) / 30) ** 2)

    def times(self, sky_dict):
        return 0.3 + np.arange(self.number_of_times) / sky_dict.f_sampling

    @parameterized.expand(NETWORKS)
    def test_qmc_sequence(self, *names):
        score = self.make_score(names)
        sequence = score._qmc_sequence
        length = 2 ** score.max_log2n_qmc
        self.assertEqual(sequence["u_tdet"].shape, (len(names), length))
        self.assertTrue(np.all((sequence["u_tdet"] >= 0) & (sequence["u_tdet"] < 1)))
        half_cell = 0.5 / score.sky_dict.f_sampling
        self.assertTrue(np.all(np.abs(sequence["t_fine"]) <= half_cell))
        self.assertTrue(np.all((sequence["psi"] >= 0) & (sequence["psi"] < np.pi)))
        # rot_psi turns the antenna patterns at polarization zero into those at psi
        fplus_fcross_0 = score.sky_dict.fplus_fcross_0[:20].astype(float)
        rotated = np.einsum("qpP,qdP->qdp", sequence["rot_psi"][:20], fplus_fcross_0)
        np.testing.assert_allclose(
            rotated, rotate_antenna_patterns(fplus_fcross_0, sequence["psi"][:20]), atol=1e-6)

    def test_chunks_and_cycling(self):
        score = self.make_score(("H1", "L1"))
        self.assertEqual(len(score._qmc_ind_chunks), 2 ** (score.max_log2n_qmc - score.log2n_qmc))
        np.testing.assert_array_equal(np.concatenate(score._qmc_ind_chunks), np.arange(2 ** score.max_log2n_qmc))
        self.assertTrue(all(len(chunk) == 2 ** score.log2n_qmc for chunk in score._qmc_ind_chunks))
        first = score._qmc_sequence
        visited = []
        for _ in range(score.n_qmc_sequences):
            score.switch_qmc_sequence()
            visited.append(score._current_qmc_sequence_id)
        self.assertEqual(visited, [1, 2, 0])
        self.assertIs(score._qmc_sequence, first)
        self.assertFalse(np.array_equal(score._qmc_sequences[1]["psi"], first["psi"]))
        with self.assertRaises(ValueError):
            self.make_score(("H1",), log2n_qmc=12, max_log2n_qmc=11)

    def test_inverse_transform(self):
        # over the whole sequence, each arrival-time index is drawn in proportion to its probability
        score = self.make_score(("H1",))
        probability = exp_normalize(np.random.default_rng(6).normal(size=(1, 50)))
        all_inds = np.arange(2 ** score.max_log2n_qmc)
        tdet_inds = score.get_tdet_inds(probability, all_inds)
        counts = np.bincount(tdet_inds[0], minlength=50)
        np.testing.assert_array_less(np.abs(counts - probability[0] * len(all_inds)), 2)
        point_mass = np.zeros((1, 50))
        point_mass[0, 17] = 1
        np.testing.assert_array_equal(score.get_tdet_inds(point_mass, all_inds[:100]), 17)

    @parameterized.expand(NETWORKS)
    def test_draw(self, *names):
        score = self.make_score(names)
        times = self.times(score.sky_dict)
        draw = score.draw_extrinsic_samples(self.proposal(len(names)), times, i_chunk=1)
        self.assertEqual(draw["n_qmc"], 2 ** score.log2n_qmc)
        number_kept = draw["physical_mask"].sum()
        self.assertGreater(number_kept, 0)
        for key in ["q_inds", "sky_inds", "sky_prior", "t_first_det", "psi"]:
            self.assertEqual(len(draw[key]), number_kept)
        np.testing.assert_array_equal(draw["q_inds"], score._qmc_ind_chunks[1][draw["physical_mask"]])
        # each sky location's delays lie within half a cell of the drawn arrival-time indices' differences
        delays_in_cells = score.sky_dict.delays[:, draw["sky_inds"]] * score.sky_dict.f_sampling
        np.testing.assert_array_less(np.abs(delays_in_cells - (draw["tdet_inds"][1:] - draw["tdet_inds"][0])),
                                     0.5 + 1e-9)
        # the arrival time at the first interferometer lies within half a cell of its drawn grid time
        np.testing.assert_array_less(np.abs(draw["t_first_det"] - times[draw["tdet_inds"][0]]),
                                     0.5 / score.sky_dict.f_sampling + 1e-12)
        np.testing.assert_array_equal(draw["psi"], score._qmc_sequence["psi"][draw["q_inds"]])
        if len(names) == 1:
            self.assertTrue(np.all(draw["physical_mask"]))
            np.testing.assert_allclose(draw["sky_prior"], 1 / score.sky_dict.f_sampling)

    def test_unphysical_proposal(self):
        # arrival times 300 cells apart at H1 and L1, beyond the largest delay (about 82 cells): no sample is kept
        score = self.make_score(("H1", "L1"))
        times = self.times(score.sky_dict)
        draw = score.draw_extrinsic_samples(self.proposal(2, centers=[300, 600]) ** 8 / np.sum(
            self.proposal(2, centers=[300, 600]) ** 8, axis=1, keepdims=True), times, i_chunk=0)
        self.assertFalse(np.any(draw["physical_mask"]))
        self.assertEqual(len(draw["sky_inds"]), 0)
        self.assertEqual(draw["tdet_inds"].shape, (2, 0))
        self.assertEqual(len(draw["t_first_det"]), 0)

class TestTimeSeriesInnerProducts(unittest.TestCase):
    """The time series of inner products, without the linear-free
    time convention, against bilby's unbinned inner products of the full waveform
    arriving at the same times."""
    linear_free = False

    @classmethod
    def setUpClass(cls):
        cls.likelihood, cls.interferometers = make_likelihood(linear_free=cls.linear_free)
        cls.polarizations = cls.likelihood.waveform_generator.frequency_domain_strain(INJECTION)
        cls.d_inner_h, cls.h_inner_h, cls.time_shift = cls.likelihood.get_time_series_inner_products(
            cls.polarizations)
        cls.full_polarizations = make_waveform_generator(
            bilby.gw.source.lal_binary_black_hole).frequency_domain_strain(INJECTION)

    def unbinned(self, interferometer, arrival_time):
        # get_detector_response with the time shift written as an offset from the start time, since an absolute GPS
        # time in float64 resolves only about 2e-7 s
        antenna_patterns = self.antenna_patterns(interferometer)
        signal = (antenna_patterns[0] * self.full_polarizations["plus"]
                  + antenna_patterns[1] * self.full_polarizations["cross"])
        signal = signal * np.exp(-2j * np.pi * interferometer.frequency_array * arrival_time)
        signal *= interferometer.frequency_mask
        return interferometer.inner_product(signal), interferometer.optimal_snr_squared(signal).real

    def antenna_patterns(self, interferometer):
        return np.array([interferometer.antenna_response(
            INJECTION["ra"], INJECTION["dec"], GEOCENT_TIME, INJECTION["psi"], mode) for mode in ["plus", "cross"]])

    def test_shapes(self):
        number_of_harmonics, number_of_edges = 4, len(self.likelihood.bin_freqs)
        times = self.likelihood.times
        self.assertEqual(self.d_inner_h.shape, (number_of_harmonics, 2, len(times), 2))
        self.assertEqual(self.h_inner_h.shape, (10, 2, 2, 2))
        self.assertEqual(self.likelihood.time_series_summary_data["H1"].shape,
                         (number_of_harmonics, len(times), number_of_edges))
        self.assertAlmostEqual(times[1] - times[0], 1 / SAMPLING_FREQUENCY)
        start_time = self.interferometers.start_time
        self.assertLess(times[0], GEOCENT_TIME - start_time - 0.1 - 0.0213)
        self.assertGreater(times[-1], GEOCENT_TIME - start_time + 0.1 + 0.0213)

    def test_against_unbinned(self):
        for index, interferometer in enumerate(self.interferometers):
            antenna_patterns = self.antenna_patterns(interferometer)
            true_arrival = GEOCENT_TIME - interferometer.strain_data.start_time + (
                interferometer.time_delay_from_geocenter(INJECTION["ra"], INJECTION["dec"], GEOCENT_TIME))
            peak = np.argmin(np.abs(self.likelihood.times - true_arrival))
            h_inner_h = np.einsum("p,q,kpq->", antenna_patterns, antenna_patterns,
                                  self.h_inner_h[..., index]).real
            for time_index in peak + np.array([-40, -4, -1, 0, 1, 4, 40]):
                time = self.likelihood.times[time_index] - self.time_shift
                d_inner_h = np.einsum("p,mp->", antenna_patterns, self.d_inner_h[:, :, time_index, index])
                expected_d_inner_h, expected_h_inner_h = self.unbinned(interferometer, time)
                # In nats, as <d|h> and <h|h> enter ln L. Here the binning error is at most 3.0e-3 nats in <d|h> and
                # 1.8e-3 nats in <h|h> (2.6e-3 nats at the peak, the parent class's own); 5e-3 nats is above that
                # and below the 1.4e-2 nats of a 6e-8 s error in the arrival time
                self.assertLess(abs(d_inner_h - expected_d_inner_h), 5e-3, msg=(interferometer.name, time_index))
                self.assertAlmostEqual(h_inner_h, expected_h_inner_h, delta=5e-3)


class TestLinearFreeTimeSeriesInnerProducts(TestTimeSeriesInnerProducts):
    """The same, with the linear-free time convention; the time series at grid
    time t is that of the waveform arriving at t minus the time shift."""
    linear_free = True

    def test_time_shift_without_linear_free_is_zero(self):
        likelihood, _ = make_likelihood(linear_free=False)
        self.assertEqual(likelihood.get_time_series_inner_products(self.polarizations)[2], 0.0)

    def test_recovers_a_known_time_shift(self):
        # the linear-free fit's own reference (the faded m = 2 plus polarization of the fiducial waveform), delayed by
        # 3 ms, is aligned by a time shift of 3 ms
        index = self.likelihood._harmonic_index(2)
        self.assertEqual(index, 0)
        delayed = {key: np.copy(value) for key, value in self.polarizations.items()}
        delayed["plus"][index] = self.likelihood._linear_free_reference * np.exp(
            -2j * np.pi * self.likelihood.bin_freqs * 3e-3)
        self.assertAlmostEqual(self.likelihood.get_linear_free_time_shift(delayed), 3e-3, delta=1e-9)


def gaussian_lnlike(d_inner_h, h_inner_h):
    """The log likelihood at the distance the waveform was generated at, standing in for the likelihood
    marginalized over distance in tests of the bookkeeping."""
    return d_inner_h - h_inner_h / 2


class TestImportanceSampling(unittest.TestCase):
    """The weights and the adaptive importance sampling, for one, two and
    three interferometers: bookkeeping against cogwheel's definitions."""
    number_of_times = 1200  # 0.146 s at 8192 Hz; longer than the largest delay range in cells (H1-V1, 435)
    m_arr = [2, 1, 3, 4]

    @classmethod
    def make_score(cls, names, lnlike=gaussian_lnlike, **kwargs):
        sky_dict = SkyDictionary(bilby.gw.detector.InterferometerList(list(names)), nsky=10 ** 5)
        return CoherentScore(sky_dict, cls.m_arr, lnlike, n_qmc_sequences=2, **kwargs)

    def times(self, sky_dict):
        return 0.3 + np.arange(self.number_of_times) / sky_dict.f_sampling

    def peaked_time_series(self, number_of_ifos, seed=3):
        # noise in every harmonic, polarization and interferometer, plus a peak at a different time in each
        rng = np.random.default_rng(seed)
        shape = (len(self.m_arr), 2, self.number_of_times, number_of_ifos)
        dh_mptd = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        for i_ifo in range(number_of_ifos):
            dh_mptd[0, 0, 600 + 20 * i_ifo, i_ifo] += 15.0
        n_pairs = len(self.m_arr) * (len(self.m_arr) + 1) // 2
        hh_mppd = np.zeros((n_pairs, 2, 2, number_of_ifos), dtype=complex)
        hh_mppd[:, 0, 0] = hh_mppd[:, 1, 1] = 1.0
        return dh_mptd, hh_mppd

    def test_time_prior_excludes_every_sample(self):
        # a time prior with zero density everywhere drops every sample
        score = self.make_score(("H1", "L1"))
        score.ln_time_prior = lambda t: np.full(np.shape(t), -np.inf)
        info = score.get_marginalization_info(*self.peaked_time_series(2), self.times(score.sky_dict))
        self.assertEqual(info.lnl_marginalized, -np.inf)
        self.assertEqual(info.n_effective, 0)
        self.assertEqual(info.n_qmc, 2 ** score.log2n_qmc)

    @parameterized.expand(NETWORKS)
    def test_weights(self, *names):
        # the estimate is the sum of the numerators over the balance-heuristic denominators
        score = self.make_score(names, lnlike=gaussian_lnlike)
        times = self.times(score.sky_dict)
        dh_mptd, hh_mppd = self.peaked_time_series(len(names))
        proposal = score.get_t_arrival_proposal(dh_mptd, hh_mppd)
        info = score.get_marginalization_info_chunk(dh_mptd, hh_mppd, times, proposal, 0)
        denominators = 2 ** score.log2n_qmc * np.prod(np.take_along_axis(proposal, info.tdet_inds, axis=1), axis=0)
        self.assertAlmostEqual(info.lnl_marginalized, logsumexp(info.ln_numerators - np.log(denominators)))
        self.assertAlmostEqual(np.sum(info.weights), 1.0)
        self.assertAlmostEqual(np.sum(info.weights_q), 1.0)
        self.assertLessEqual(len(info.weights_q), len(np.unique(info.q_inds)))
        self.assertTrue(np.all(np.isfinite(info.d_h)))
        second = score.get_marginalization_info_chunk(dh_mptd, hh_mppd, times, proposal, 1)
        both = MarginalizationInfo(**{key: getattr(info, key) for key in [
            "qmc_sequence_id", "ln_numerators", "q_inds", "o_inds", "sky_inds", "t_first_det", "d_h", "h_h",
            "tdet_inds", "flip_psi", "proposals_n_qmc", "proposals"]})
        both.update(second)
        self.assertEqual(both.n_qmc, 2 ** (score.log2n_qmc + 1))
        self.assertEqual(len(both.q_inds), len(info.q_inds) + len(second.q_inds))
        # with the same proposal twice, the estimate is the mean of the two chunks' estimates
        self.assertAlmostEqual(both.lnl_marginalized,
                               np.logaddexp(info.lnl_marginalized, second.lnl_marginalized) - np.log(2))

    @parameterized.expand(NETWORKS)
    def test_kde_proposal(self, *names):
        score = self.make_score(names, lnlike=gaussian_lnlike)
        times = self.times(score.sky_dict)
        dh_mptd, hh_mppd = self.peaked_time_series(len(names))
        proposal = score.get_t_arrival_proposal(dh_mptd, hh_mppd)
        info = score.get_marginalization_info_chunk(dh_mptd, hh_mppd, times, proposal, 0)
        kde = score.kde_t_arrival_prob(info, times)
        self.assertEqual(kde.shape, proposal.shape)
        np.testing.assert_allclose(kde.sum(axis=1), 1)
        self.assertTrue(np.all(kde >= 0))

    def test_adaptation_stops(self):
        dh_mptd, hh_mppd = self.peaked_time_series(2)
        # an unreachable effective sample size: every chunk of the sequence is used
        score = self.make_score(("H1", "L1"), lnlike=gaussian_lnlike, min_n_effective=1e9)
        times = self.times(score.sky_dict)
        info = score.get_marginalization_info(dh_mptd, hh_mppd, times)
        self.assertEqual(info.n_qmc, 2 ** score.max_log2n_qmc)
        self.assertEqual(len(info.proposals), len(score._qmc_ind_chunks))
        self.assertIsNone(score._t_arrival_prob)  # not remembered: the effective sample size stays below its minimum
        # an estimate below the threshold: the loop stops at the first chunk after which the effective sample size
        # exceeds 2
        info = score.get_marginalization_info(dh_mptd, hh_mppd, times, lnl_marginalized_threshold=np.inf)
        self.assertGreater(info.n_effective, 2)
        self.assertLess(info.n_qmc, 2 ** score.max_log2n_qmc)
        last_chunk = info.q_inds >= info.n_qmc - 2 ** score.log2n_qmc
        if info.n_qmc > 2 ** score.log2n_qmc:  # without the last chunk the effective sample size was at most 2
            earlier = MarginalizationInfo(
                qmc_sequence_id=info.qmc_sequence_id, ln_numerators=info.ln_numerators[~last_chunk],
                q_inds=info.q_inds[~last_chunk], o_inds=info.o_inds[~last_chunk],
                sky_inds=info.sky_inds[~last_chunk], t_first_det=info.t_first_det[~last_chunk],
                d_h=info.d_h[~last_chunk], h_h=info.h_h[~last_chunk], tdet_inds=info.tdet_inds[:, ~last_chunk],
                flip_psi=info.flip_psi[~last_chunk], proposals_n_qmc=info.proposals_n_qmc[:-1],
                proposals=info.proposals[:-1])
            self.assertLessEqual(earlier.n_effective, 2)
        # a reachable effective sample size: one chunk, and the proposal is remembered
        score = self.make_score(("H1", "L1"), lnlike=gaussian_lnlike, min_n_effective=1)
        info = score.get_marginalization_info(dh_mptd, hh_mppd, times)
        self.assertEqual(len(info.proposals), 1)
        np.testing.assert_allclose(score._t_arrival_prob, info.proposals[-1])

    def test_optimize_beta_temperature(self):
        score = self.make_score(("H1",), lnlike=gaussian_lnlike)
        dh_mptd, hh_mppd = self.peaked_time_series(1)
        score.optimize_beta_temperature(dh_mptd, hh_mppd, self.times(score.sky_dict))
        self.assertEqual(score.beta_temperature.shape, (1,))
        self.assertTrue(0.1 <= score.beta_temperature[0] <= 1)
        self.assertEqual(score.min_n_effective, 50)


class TestTimeGridPadding(unittest.TestCase):
    """The padding of the time grid and the check of the linear-free time shift against its margin."""

    @staticmethod
    def make(time_padding=None):
        cls = bilby.gw.likelihood.ExtrinsicMarginalizedRelativeBinningGravitationalWaveTransient
        interferometers = bilby.gw.detector.InterferometerList(["H1", "L1"])
        interferometers.set_strain_data_from_zero_noise(
            sampling_frequency=SAMPLING_FREQUENCY, duration=DURATION, start_time=GEOCENT_TIME - DURATION + 2)
        priors = dict(geocent_time=bilby.core.prior.Uniform(GEOCENT_TIME - 0.1, GEOCENT_TIME + 0.1))
        likelihood = cls.__new__(cls)  # only the time grid is needed
        likelihood.times, likelihood.time_padding = cls._get_time_grid(
            interferometers, priors, make_waveform_generator(bilby.gw.source.lal_binary_black_hole), time_padding)
        likelihood._largest_time_shift_warned = None
        return likelihood

    def test_default_padding_and_margin(self):
        likelihood = self.make()
        widened = 0.2 + 2 * bilby.core.utils.radius_of_earth / bilby.core.utils.speed_of_light
        self.assertAlmostEqual(likelihood.time_padding, widened / 10)
        self.assertAlmostEqual(likelihood.time_shift_margin, 0.04 * widened, delta=1e-3)

    def test_check_time_shift(self):
        likelihood = self.make()
        self.assertTrue(likelihood.check_time_shift(0.5 * likelihood.time_shift_margin))
        with self.assertLogs("bilby", level="WARNING") as logs:
            self.assertFalse(likelihood.check_time_shift(-0.03))
        self.assertIn("time_padding", logs.output[0])
        # a smaller shift beyond the margin is not warned about again
        with self.assertNoLogs("bilby", level="WARNING"):
            self.assertFalse(likelihood.check_time_shift(0.02))
        # the padding the warning names covers the shift
        padded = self.make(time_padding=likelihood.required_time_padding(0.03))
        self.assertAlmostEqual(padded.time_shift_margin, 0.03, delta=1e-3)
        self.assertTrue(padded.check_time_shift(0.0299))


if __name__ == "__main__":
    unittest.main()
