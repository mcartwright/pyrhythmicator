# -*- coding: utf-8 -*-
from __future__ import division

import copy
import datetime
import os
import random
import re

import jams
import librosa
import numpy as np
import pandas as pd
import scipy.io.wavfile
import scipy.stats

from pyrhythmicator_exceptions import PyrhythmicatorError


def _write_wav(path, y, sample_rate, norm=True, dtype='int16'):
    """
    Write .wav file to disk.

    Parameters
    ----------
    path : str
        File path to write wav file
    y : np.array
        Audio signal array
    sample_rate : float
    norm : bool
        Peak-normalize `y` before writing to disk.
    dtype : str
        This numpy array type will dictate what the sample format of the audio file is.

    Returns
    -------
    None
    """
    if norm:
        y /= np.max(np.abs(y))
    scipy.io.wavfile.write(path, sample_rate, np.array((y * (np.iinfo(dtype).max - 1))).astype(dtype))


def _dict_of_array_to_dict_of_list(d):
    new_dict = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            new_dict[k] = v.tolist()
        else:
            new_dict[k] = v
    return new_dict


def _dict_of_list_to_dict_of_array(d):
    new_dict = {}
    for k, v in d.items():
        if isinstance(v, list):
            new_dict[k] = np.array(v)
        else:
            new_dict[k] = v
    return new_dict


def _repeat_annotations(ann, repetitions):
    frames = [ann.data]

    for i in range(1, repetitions):
        frame = copy.deepcopy(ann.data)
        frame.time += datetime.timedelta(seconds=(ann.duration * i))
        frames.append(frame)

    frame = pd.DataFrame(pd.concat(frames, ignore_index=True))
    ann.data = jams.JamsFrame.from_dataframe(frame)
    ann.duration *= repetitions
    return ann


def _rotate_annotations(ann, rotation_sec):
    dur = datetime.timedelta(seconds=ann.duration)
    ann.data.time += datetime.timedelta(seconds=rotation_sec)
    ann.data.ix[ann.data.time >= dur, 'time'] -= dur
    ann.data = jams.JamsFrame.from_dataframe(ann.data.sort_values('time').reset_index(drop=True))
    return ann


def _trim_annotations(ann, min_sec, max_sec):
    ann.data.time -= datetime.timedelta(seconds=min_sec)
    max_sec -= min_sec
    ann.data = ann.data.ix[(ann.data.time >= datetime.timedelta(seconds=0)) &
                           (ann.data.time <= datetime.timedelta(seconds=max_sec))]
    dur = max(max_sec - min_sec, ann.data.time.max().total_seconds())
    ann.duration = dur
    ann.data = jams.JamsFrame.from_dataframe(ann.data)
    return ann


def sort_audio_by_centroid(audio_files, sample_rate=44100):
    """
    Sort audio files by centroid so we know how to functionally order them

    Parameters
    ----------
    audio_files
    sample_rate

    Returns
    -------
    None
    """
    centroids = []
    for af in audio_files:
        sample, _ = librosa.load(af, sample_rate, mono=True)
        centroids.append(np.median(librosa.feature.spectral_centroid(sample, sample_rate)))
    print([audio_files[i] for i in np.argsort(centroids)])


def list_audio_files_in_dir(directory, extensions=('.wav', '.mp3', '.aif', '.aiff'), prepend_path=False):
    """
    Populate a list with all of the audio files in a directory.

    Parameters
    ----------
    directory : str
    extensions : list[str]
        The audio file extensions to search for
    prepend_path : bool
        Prepend the file path in front of the audio files.
    Returns
    -------
    audio_files : list[str]
    """
    audio_files = [f for f in os.listdir(directory) if os.path.splitext(f)[1] in extensions]

    if prepend_path:
        audio_files = [os.path.join(directory, f) for f in audio_files]

    return audio_files


def strat_level_to_note_dur_frac(divisor, dotted):
    """
    Return the numerator and denominator of pulse length (in whole notes)

    Parameters
    ----------
    divisor : int
    dotted : bool

    Returns
    -------
    num : int
    denom : int
    """
    if dotted:
        num = 3
        denom = divisor * 2
    else:
        num = 1
        denom = divisor
    return num, denom


def log_to_linear_amp(x, arange=(-48., 0.)):
    """
    Convert a 0-1 log-scaled signal (whose 0 and 1 levels are defined by `arange`) to linear scale.

    Parameters
    ----------
    x : np.array
        Input signal that ranges from 0. to 1.
    arange : tuple[float]
        The range of the input in dB

    Returns
    -------
    x_linear : np.array
        Linear-scaled x

    Examples
    --------
    >>> log_to_linear_amp(np.array([1.]))
    array([ 1.])

    >>> log_to_linear_amp(np.array([0.5]), arange=(-6., 0.))
    array([ 0.70794578])

    >>> log_to_linear_amp(np.array([0.]), arange=(-6., 0.))
    array([ 0.])

    >>> log_to_linear_amp(0., arange=(-6., 0.))
    0.0
    """
    x_linear = x * (arange[1] - arange[0]) + arange[0]
    x_linear = (10.0**(x_linear/20.)) * (x > 0.)  # make sure 0 is still 0
    return x_linear


def calc_pulse_length(strat_level, tempo):
    """
    Calculate the length of a pulse in seconds given a stratification level and the tempo

    Parameters
    ----------
    strat_level : str
    tempo : float

    Returns
    -------
    pulse_length : float
        Pulse length in seconds

    Examples
    --------
    >>> calc_pulse_length('4n', 60.)
    1.0

    >>> calc_pulse_length('4n', 120.)
    0.5

    >>> calc_pulse_length('8n', 120.)
    0.25

    >>> calc_pulse_length('4nd', 60.)
    1.5
    """
    strat_num, strat_denom = strat_level_to_note_dur_frac(*parse_strat_level(strat_level))

    # (minute / beat) * (seconds / minute) * (beats / whole_note) * (whole_note / pulse)
    pulse_length = (1.0 / tempo) * 60 * 4 * strat_num / strat_denom
    return pulse_length


def calc_metric_durations(strat_level, num_pulses, tempo, sample_rate):
    """
    Calculate the duration of a pulse and bar in seconds and samples

    Parameters
    ----------
    strat_level : str
    num_pulses : int
    tempo : float
    sample_rate : float

    Returns
    -------
    pulse_length_sec : float
    pulse_length_samples : int
    bar_length_sec : float
    bar_length_samples : int
    """
    pulse_length_sec = calc_pulse_length(strat_level, tempo)
    pulse_length_samples = pulse_length_sec * sample_rate
    bar_length_samples = int(round(pulse_length_samples * num_pulses))
    bar_length_sec = bar_length_samples / sample_rate
    return pulse_length_sec, pulse_length_samples, bar_length_sec, bar_length_samples


def parse_strat_level(strat_level):
    """
    Parse the stratification level

    Parameters
    ----------
    strat_level : str

    Returns
    -------
    divisor : int
    dotted : bool

    Raises
    ------
    PyrhythmicatorError
        If `strat_level` is not in the valid form.

    Examples
    --------
    >>> parse_strat_level('128nd')
    (128, True)

    >>> parse_strat_level('64n')
    (64, False)

    >>> parse_strat_level('94n')
    Traceback (most recent call last):
     ...
    ValueError: Argument `level` is an invalid form

    >>> parse_strat_level('4ndnd')
    Traceback (most recent call last):
     ...
    ValueError: Argument `level` is an invalid form
    """
    pat = "^([0-9]+)([n,d]+)$"
    match = re.match(pat, strat_level)
    if match is None:
        raise PyrhythmicatorError('Argument `strat_level` is an invalid form')

    divisor = match.group(1)
    if match.group(2) == 'nd':
        dotted = True
    elif match.group(2) == 'n':
        dotted = False
    else:
        raise PyrhythmicatorError('Argument `strat_level` is an invalid form')

    if divisor not in ('1', '2', '4', '8', '16', '32', '64', '128'):
        raise PyrhythmicatorError('Argument `strat_level` is an invalid form')

    return int(divisor), dotted


def _validate_strat_level(ts_num, ts_denom, strat_num, strat_denom):
    """
    Check if the stratification level is valid for the time signature

    Parameters
    ----------
    ts_num : int
        Time signature numerator
    ts_denom : int
        Time signature denominator
    strat_num : int
        Stratification level numerator
    strat_denom : int
        Stratification level denominator

    Returns
    -------
    isvalid : bool

    Examples
    --------
    >>> _validate_strat_level(3, 4, *strat_level_to_note_dur_frac(4, False))
    True

    >>> _validate_strat_level(3, 4, *strat_level_to_note_dur_frac(4, True))
    True

    >>> _validate_strat_level(3, 2, *strat_level_to_note_dur_frac(1, False))
    False

    >>> _validate_strat_level(3, 2, *strat_level_to_note_dur_frac(1, True))
    True

    >>> _validate_strat_level(3, 2, *strat_level_to_note_dur_frac(2, True))
    True

    >>> _validate_strat_level(3, 4, *strat_level_to_note_dur_frac(128, False))
    True

    >>> _validate_strat_level(16, 8, *strat_level_to_note_dur_frac(128, True))
    False

    >>> _validate_strat_level(3, 4, *strat_level_to_note_dur_frac(4, False))
    True
    """
    isvalid = ((ts_num * strat_denom) % (strat_num * ts_denom)) == 0

    return isvalid


ALL_STRATS = [str(v) + d for v in (2 ** np.arange(8)).tolist() for d in ['n', 'nd']]


def get_valid_strat_levels(ts_num, ts_denom):
    """
    Return the valid possible strat levels

    Parameters
    ----------
    ts_num
    ts_denom

    Returns
    -------
    possible_strats : list[str]
    """
    possible_strats = [sl for sl in ALL_STRATS
                       if _validate_strat_level(ts_num,
                                                ts_denom,
                                                *strat_level_to_note_dur_frac(*parse_strat_level(sl))) and
                       len(stratify(ts_num, ts_denom, sl)) > 0]
    return possible_strats


def random_strat_level(ts_num, ts_denom):
    """
    Return a random valid stratification level given `ts_num` and `ts_denom`

    Parameters
    ----------
    ts_num : int
        Time-signature numerator
    ts_denom : int
        Time-signature denominator

    Returns
    -------
    str
    """
    possible_strats = get_valid_strat_levels(ts_num, ts_denom)
    return random.choice(possible_strats)


def _validate_meter(num, denom):
    """
    Make sure both ints and that denominator is 2^n

    Parameters
    ----------
    num : int
        Time signature numerator
    denom : int
        Time signature denominator

    Returns
    -------
    isvalid : bool

    Examples
    --------
    >>> _validate_meter(4, 32)
    True

    >>> _validate_meter(4, 7)
    False

    >>> _validate_meter(8, 2)
    True

    >>> _validate_meter(8, 9)
    False
    """
    isvalid = isinstance(num, int) and isinstance(denom, int) and ((np.log2(denom) % 1) == 0)

    return isvalid


def calc_num_pulses(ts_num, ts_denom, strat_level):
    """
    Calculate the number of pulses per measure given a time signature and stratification level

    Parameters
    ----------
    ts_num : int
        Numerator of time signature
    ts_denom : int
        Denominator of time signature
    strat_level : str
        Stratification level, e.g. 1n = whole note, 2n = half note, 1nd = dotted whole note, etc.
        (valid values up to 128)

    Returns
    -------
    num_pulses : int
        The number of pulses per measure
    """
    # parse metric
    strat_divisor, strat_dotted = parse_strat_level(strat_level)
    strat_num, strat_denom = strat_level_to_note_dur_frac(strat_divisor, strat_dotted)

    # check validity
    if not _validate_meter(ts_num, ts_denom):
        raise PyrhythmicatorError('Invalid time signature')

    if not _validate_strat_level(ts_num, ts_denom, strat_num, strat_denom):
        raise PyrhythmicatorError('Invalid stratification level (`strat_level`)')

    # num pulses
    num_pulses = (ts_num * strat_denom) // (strat_num * ts_denom)

    return num_pulses


def stratify(ts_num, ts_denom, strat_level):
    """
    Stratifies the meter into metrical levels

    Parameters
    ----------
    ts_num : int
        Numerator of time signature
    ts_denom : int
        Denominator of time signature
    strat_level : str
        Stratification level, e.g. 1n = whole note, 2n = half note, 1nd = dotted whole note, etc.
        (valid values up to 128)

    Returns
    -------
    prime_factors : list[int]
        List of prime numbers that describes how each metrical level divides the previous one.

    Raises
    ------
    PyrhythmicatorError
        If `strat_level` is invalid. Or if the meter (`ts_num` and `ts_denom`) is invalid.

    Examples
    --------
    >>> stratify(3, 4, '32n')
    [3, 2, 2, 2]

    >>> stratify(3, 4, '4n')
    [3]

    >>> stratify(3, 4, '64nd')
    [2, 2, 2, 2, 2]

    >>> stratify(5, 8, '64n')
    [5, 2, 2, 2]

    >>> stratify(12, 8, '16n')
    [2, 2, 3, 2]

    >>> stratify(18, 8, '16n')
    [2, 3, 3, 2]

    >>> stratify(18, 8, '8nd')
    [2, 3, 2]

    >>> stratify(18, 8, '8n')
    [2, 3, 3]

    >>> stratify(60, 8, '8n')
    [2, 2, 5, 3]

    >>> stratify(60, 8, '8nd')
    [5, 2, 2, 2]

    >>> stratify(30, 8, '8n')
    [2, 3, 5]

    >>> stratify(18, 16, '32nd')
    [2, 3, 2, 2]

    >>> stratify(3, 4, '16n')
    [3, 2, 2]

    >>> stratify(6, 8, '16n')
    [2, 3, 2]
    """
    num_pulses = calc_num_pulses(ts_num, ts_denom, strat_level)

    # find prime factors
    prime_factors = _find_prime_factors(num_pulses)

    # swap 3s if numerator is divisible by 2, and there is a 3
    if ((ts_num % 2) == 0) and (3 in prime_factors):
        num_prime_factors = _find_prime_factors(ts_num)
        count = np.sum(np.array(num_prime_factors) == 2)

        # swap depending on placement
        for i in range(count):
            for j in range(len(prime_factors)):
                if prime_factors[j] == 2 and (j > i):
                    a = prime_factors[j]
                    b = prime_factors[i]
                    prime_factors[i] = a
                    prime_factors[j] = b

    return prime_factors


def _find_prime_factors(n):
    """
    Find all the prime factors of `n`, sorted from largest to smallest

    Parameters
    ----------
    n : int
        Number to factorize

    Returns
    -------
    factors : list[int]

    Notes
    -----
    From http://stackoverflow.com/questions/15347174/python-finding-prime-factors

    Examples
    --------
    >>> _find_prime_factors(8)
    [2, 2, 2]

    >>> _find_prime_factors(25)
    [5, 5]

    >>> _find_prime_factors(26)
    [13, 2]

    >>> _find_prime_factors(29)
    [29]

    >>> _find_prime_factors(14)
    [7, 2]

    >>> _find_prime_factors(12)
    [3, 2, 2]
    """
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    factors = sorted(factors, reverse=True)
    return factors


def calc_basic_indispensability(p, n):
    """
    >>> calc_basic_indispensability(3, 0)
    2

    >>> calc_basic_indispensability(2, 1)
    0

    >>> calc_basic_indispensability(5, 4)
    2

    >>> calc_basic_indispensability(7, 5)
    2

    >>> calc_basic_indispensability(13, 4)
    10
    """
    table = {
        2: [1, 0, ],
        3: [2, 0, 1, ],
        5: [4, 0, 1, 3, 2, ],
        7: [6, 0, 4, 1, 5, 2, 3, ],
        11: [10, 0, 6, 3, 9, 1, 7, 2, 8, 4, 5, ],
        13: [12, 0, 7, 3, 10, 1, 8, 4, 11, 2, 9, 5, 6, ],
        17: [16, 0, 9, 4, 13, 2, 11, 6, 15, 1, 10, 5, 14, 3, 12, 7, 8, ],
        19: [18, 0, 10, 3, 13, 6, 16, 1, 11, 4, 14, 7, 17, 2, 12, 5, 15, 8, 9, ],
        23: [22, 0, 12, 6, 18, 3, 15, 9, 21, 1, 13, 7, 19, 2, 14, 8, 20, 4, 16, 5, 17, 10, 11, ],
        29: [28, 0, 15, 7, 22, 4, 19, 11, 26, 1, 16, 8, 23, 5, 20, 12, 27, 2, 17, 9, 24, 3, 18, 10, 25, 6, 21, 13, 14, ],
        31: [30, 0, 16, 5, 21, 10, 26, 3, 19, 8, 24, 13, 29, 1, 17, 6, 22, 11, 27, 2, 18, 7, 23, 12, 28, 4, 20, 9, 25, 14, 15, ],
    }

    return table[p][n]


def calc_indispensabilities(num_pulses, num_levels, prime_factors):
    """
    Calculate the indispensability value from C. Barlow's formula

    Parameters
    ----------
    num_pulses : int
    num_levels : int
    prime_factors : list[int]

    Returns
    -------
    indisp_by_strata : np.array

    References
    ----------
    Barlow, C. "Two essays on theory". Computer Music Journal, 11, 44-60, 1987

    Examples
    --------
    >>> np.sum(calc_indispensabilities(6, 2, [3, 2]), axis=0).tolist()
    [5, 0, 3, 1, 4, 2]

    >>> np.sum(calc_indispensabilities(3, 1, [3,]), axis=0).tolist()
    [2, 0, 1]

    >>> np.sum(calc_indispensabilities(8, 3, [2, 2, 2]), axis=0).tolist()
    [7, 0, 4, 2, 6, 1, 5, 3]

    """
    m = [(num_pulses + i - 1) % num_pulses for i in range(num_pulses)]
    indisp_by_strata = np.zeros([num_levels, num_pulses], dtype=int)

    for i in range(num_levels):
        q_i = 1
        q_z = 1
        q_zr = prime_factors[num_levels - i - 1]

        for k in range(num_levels - i):
            if k != 0:
                q_i *= prime_factors[k-1]

        for k in range(i+1):
            idx = num_levels - k
            if idx != num_levels:
                q_z *= prime_factors[idx]

        for j in range(num_pulses):
            calc = (1 + (m[j] // q_z)) % q_zr
            b_i = calc_basic_indispensability(int(q_zr), int(calc))
            indisp_by_strata[i, j] = q_i * b_i

    return indisp_by_strata


def calc_weights(stratification, range_factor):
    """
    Calculate the weights given a stratification

    Parameters
    ----------
    stratification
    range_factor

    Returns
    -------
    weights : np.array
    indisp : np.array

    Examples
    --------
    >>> calc_weights(stratify(3, 4, '4n'), 0.5)
    (array([ 1.        ,  0.66666667,  0.83333333]), array([2, 0, 1]))

    >>> calc_weights(stratify(3, 4, '4n'), 1.0)
    (array([ 1.,  1.,  1.]), array([2, 0, 1]))

    >>> calc_weights(stratify(3, 4, '4n'), 0.0)
    (array([ 1.        ,  0.33333333,  0.66666667]), array([2, 0, 1]))

    >>> calc_weights(stratify(6, 8, '8nd'), 0.5)
    (array([ 1.   ,  0.375,  0.75 ,  0.5  ]), array([3, 0, 2, 1]))

    >>> calc_weights(stratify(9, 4, '16nd'), 0.3)
    (array([ 1.        ,  0.009675  ,  0.0375    ,  0.019125  ,  0.16      ,
            0.0144    ,  0.069     ,  0.02385   ,  0.53333333,  0.01125   ,
            0.048     ,  0.0207    ,  0.23      ,  0.015975  ,  0.0795    ,
            0.025425  ,  0.76666667,  0.012825  ,  0.0585    ,  0.022275  ,
            0.3       ,  0.01755   ,  0.09      ,  0.027     ]), array([23,  0, 12,  6, 18,  3, 15,  9, 21,  1, 13,  7, 19,  4, 16, 10, 22,
            2, 14,  8, 20,  5, 17, 11]))

    """
    num_levels = len(stratification)
    num_pulses = int(np.prod(stratification))
    group = stratification
    weights = np.zeros(num_pulses)

    indisp_by_strata = calc_indispensabilities(num_pulses, num_levels, stratification)
    indisp = np.sum(indisp_by_strata, axis=0)  # sum over levels

    k = num_pulses - 1
    i_temp = 1.0
    i_counter = 0

    if range_factor > 1:
        ro = (range_factor - 1.0) / (range_factor - (range_factor**(1.0 - num_levels)))
    else:
        ro = 1.0 / num_levels

    norm = 1.
    for i in range(num_levels):
        i_temp *= group[i]
        i_beats = int(i_temp - i_counter)

        if (range_factor <= 1) and (range_factor >= 0):
            _range = (range_factor ** i) - (range_factor ** (i + 1))
            dif = _range / i_beats
        elif range_factor > 1.:
            _range = ro / (range_factor ** i)
            dif = _range / i_beats
        else:  # if range factors < 0.
            _range = ro
            dif = _range / i_beats

        for j in range(i_beats):
            p = 0
            while (p < num_pulses) and (indisp[p] != k):
                p += 1
            # replace with p = indisp.index(k)
            weights[p] = norm - (dif * j)
            k -= 1
            i_counter += 1

        norm -= _range

    return weights, indisp


class Sequencer(object):
    """
    A port of Sioros and Guedes's kin.sequencer Max object.

    A set of probabilities is used to trigger stochastic events. A density and an exponential factor let the user
    control further these probabilities. The external can also syncopate in real time by anticipating events.

    Attributes
    ----------
    weights : np.array
        The weights for each metric position in a measure.
    metric_factor : float
        An exponential factor controls the strength of the metrical feel in the resulted performance. When zero, all
        pulses have the equal probability of triggering an event. When equals one, the probabilities are equal to the
        input list. Default is 1.
    syncopate_factor : float
        Syncopation is generated by anticipating events. The higher this factor, the higher  the probability of
        anticipating an event. Ranges between 0. and  1. Default is 0.
    density : float
        Controls the density of events per cycle. Zero means no events get triggered. One means maximum density, which
        depends on the input probabilities. Default is 1.
    event_var : float
        General variation of the triggered events (e.g. calling  and the second the variation in syncopation, by
        anticipating different events in each cycle


    """
    MIN_PROB_VALUE = 0.02

    def __init__(self,
                 weights,
                 metric_factor=1.0,
                 syncopate_factor=0.0,
                 density=1.,
                 event_var=1.,
                 sync_var=1.,
                 weight_minimum=0):
        self.weights = weights
        self.metric_factor = metric_factor
        self.syncopate_factor = syncopate_factor
        self.density = max(0, density)
        self.event_var = event_var
        self.sync_var = sync_var
        self.weight_minimum = weight_minimum

        self.num_pulses = len(self.weights)
        self.rescaled_weights = self.weights
        self.prob_factor = None
        self.total_prob = None
        self.orig_total_prob = None
        self.scores = None
        self.sync_score = None
        self.sync_stop_score = None
        self.counter = 0
        self.sync_count = 0
        self.previous_triggered = False
        self.sync_direction = np.zeros(self.num_pulses, dtype='int')
        self.outputs = np.zeros(self.num_pulses)
        self.pattern = np.zeros(self.num_pulses)

        self.reset_all_scores()
        self._rescale_weights()
        self._calc_orig_total_prob()
        self._calc_total_prob()
        self._calc_prob_factor()

    def _rescale_weights(self):
        """
        Rescale weights so that minimum is now 0, e.g (W - m) / (1 - m)

        Returns
        -------
        None

        Examples
        --------
        >>> s = Sequencer(np.array([0.3,0.5,1.0]), weight_minimum=0.5)
        >>> s.rescaled_weights
        array([ 0.,  0.,  1.])

        >>> s = Sequencer(np.array([0.3,0.5,1.0]), weight_minimum=0.3)
        >>> s.rescaled_weights[0]==0, s.rescaled_weights[1]>0, s.rescaled_weights[2]==1
        (True, True, True)
        """
        if self.weight_minimum == 1:
            self.rescaled_weights = np.zeros_like(self.weights)
        else:
            self.rescaled_weights = (self.weights - self.weight_minimum) / (1.0 - self.weight_minimum)
            self.rescaled_weights = np.clip(self.rescaled_weights, 0., 1.)

    def _calc_orig_total_prob(self):
        if self.num_pulses <= 0:
            return

        total_prob = 0
        for i in range(self.num_pulses):
            if self.weights[i] > 1:
                total_prob += 1
            elif self.weights[i] > 0:
                total_prob += self.weights[i]

        self.orig_total_prob = total_prob

    def _calc_total_prob(self):
        """
        Calculates the total probability of triggering a note in a whole measure

        Returns
        -------
        None
        """
        self.total_prob = 0.
        for i in range(self.num_pulses):
            if self.weights[i] > 1:
                self.total_prob += 1
            elif self.weights[i] >= self.MIN_PROB_VALUE:
                self.total_prob += self.weights[i]**self.metric_factor
            else:  # self.weights[i] < self.MIN_PROB_VALUE
                self.total_prob += self.MIN_PROB_VALUE**self.metric_factor

    def _calc_prob_factor(self):
        """
        Calculates the probability factor used in the trigger_step method
        (depends on the user controlled `density` factor)

        Returns
        -------
        None
        """
        if self.total_prob == 0:
            self.prob_factor = 0
        else:
            self.prob_factor = self.density * self.orig_total_prob / self.total_prob

    def reset_all_scores(self):
        self.scores = np.random.random([self.num_pulses, 2])
        self.sync_score = np.random.random([self.num_pulses, 2])
        self.sync_stop_score = np.random.random([self.num_pulses, 2])

    @staticmethod
    def _variation_score(score, score_range):
        """
        Returns a random number between -0.5 range ~ 0.5 range

        Parameters
        ----------
        score : float
        score_range : float

        Returns
        -------
        score_variation : float
        """
        if (score + score_range*0.5) > 1:
            return 1 - (random.random() * score_range) - score
        elif (score - score_range * 0.5) < 0:
            return score_range * random.random() - score
        else:
            return (random.random() - 0.5) * score_range

    def trigger_step(self, count=None):
        """
        Calculate whether or not a onset should occur at the current step.

        Parameters
        ----------
        count : int
            Current step

        Returns
        -------
        None
        """
        if count is not None:
            self.counter = count
        self.scores[self.counter, 1] = self._variation_score(self.scores[self.counter, 0], self.event_var)
        self.sync_score[self.counter, 1] = self._variation_score(self.sync_score[self.counter, 0], self.sync_var)
        self.sync_stop_score[self.counter, 1] = self._variation_score(self.sync_stop_score[self.counter, 0],
                                                                      self.sync_var)

        idx = self._sequential_syncopation()
        if (idx >= 0) and (idx < self.num_pulses):
            self.outputs[self.counter] = self.step_prob(idx)
        else:
            self.outputs[self.counter] = 0

        if self.outputs[self.counter] > 1:
            self.outputs[self.counter] = 1

        scr = np.sum(self.scores[self.counter])
        if self.outputs[self.counter] > scr:
            idx = self.counter + self.sync_direction[self.counter]
            while idx >= self.num_pulses:
                idx -= self.num_pulses
            while idx < 0:
                idx += self.num_pulses
            self.pattern[self.counter] = self.weights[idx]
            self.previous_triggered = True
        else:
            self.previous_triggered = False

    def _sequential_syncopation(self):
        """
        Determine whether to syncopate this beat. Returns the index of the pulse to be played back applying the rules
        for sequential syncopation if a negative is return then NO trigger is forced irrelevant of index

        Returns
        -------
        idx : int
            Index of hte pulse to be played back
        """
        sync_score = np.sum(self.sync_score[self.counter])
        sync_stop_score = np.sum(self.sync_stop_score[self.counter])

        if sync_score > 1:
            sync_score = 2 - sync_score
        elif sync_score < 0:
            sync_score = -sync_score

        if sync_stop_score > 1:
            sync_stop_score -= int(sync_stop_score)
        elif sync_stop_score < 0:
            sync_stop_score = -sync_stop_score

        idx = self.counter
        i = idx - 1
        if i < 0:
            i += self.num_pulses

        self.sync_count = 0

        while (self.sync_direction[i] != 0) and (self.sync_count < self.num_pulses):
            i -= 1
            if i < 0:
                i += self.num_pulses
            self.sync_count += 1

        sync_stop_factor = 2 * self.sync_count + 2 * self.sync_var * self.sync_count
        sync_stop_prob = (sync_stop_factor / self.num_pulses) * (self.weights[self.counter]**0.25)
        if ((self.sync_count % 2) == 1) or \
                ((abs(self.syncopate_factor) > sync_score) and (sync_stop_prob < sync_stop_score)):
            self.sync_direction[self.counter] = 1
            idx += 1
            while idx < 0:
                idx += self.num_pulses

            while idx >= self.num_pulses:
                idx -= self.num_pulses

            self.sync_count += 1
        else:
            self.sync_direction[self.counter] = 0
            i = self.counter - 1
            if i < 0:
                i += self.num_pulses
            if (self.sync_count != 0) and (self.sync_count < 3) and self.previous_triggered:
                idx = -1
            self.sync_count = 0

        return idx

    def step_prob(self, idx):
        """
        Calculates and returns the probability of specific index

        Parameters
        ----------
        idx

        Returns
        -------
        prob : float
            Probability of index `idx`
        """
        if (self.rescaled_weights[idx] < self.MIN_PROB_VALUE) and (self.metric_factor < 1.):
            return self.prob_factor * (self.MIN_PROB_VALUE**self.metric_factor)
        else:
            return self.prob_factor * (self.rescaled_weights[idx]**self.metric_factor)

    def create_pattern(self, reset_scores=True):
        """
        Create a rhythm pattern.

        Parameters
        ----------
        reset_scores : bool
            Do a reset of scores, so this pattern is not at all dependent on the previous call to create_pattern.

        Returns
        -------
        output : dict
            'beat_pattern' : np.array
                Length of self.num_pulses, where each non-zero value denotes an onset and the weight of that beat.
            'beat_probability' : np.array
                Length of self.num_pulses, where each value is the probability used to calculate if the beat has an
                onset.
            'sync_pattern' : np.array
                Length of self.num_pulses, where each element can be 0 or 1. 1 denotes whether that beat is syncopated,
                anticipating the next beat.
            'density' : float
                The average beat_probability.

        """
        if reset_scores:
            self.reset_all_scores()
        self.pattern = np.zeros(self.num_pulses)
        self.outputs = np.zeros(self.num_pulses)
        for i in range(self.num_pulses):
            self.trigger_step(i)

        output = {'beat_pattern': self.pattern,
                  'beat_probability': self.outputs,
                  'sync_pattern': self.sync_direction,
                  'avg_beat_prob': self.orig_total_prob / self.num_pulses}
        return output


class PatternGenerator(object):
    """
    Synthesize percussive audio loops using high level parameters and specified audio sample files.

    Examples
    --------
    >>> pat_gen = PatternGenerator(4, 4, '16n')
    >>> pat_gen.synthesize('output_file.wav', audio_files=['../test/data/01_kick.wav','../test/data/02.snare.wav'])

    Attributes
    ----------
    ts_num : int
        Time signature numerator
    ts_denom : int
        Time signature denominator
    strat_level : list[str]
        Stratification level (pulse/tatum) of each pattern, e.g. '8n' for eigth-note, '4nd' for dotted quarter note.
        Up to '128'
    metric_factor : list[float]
        How metrical the rhythm is for each pattern (0. = not metrical at all, 1. = very metrical)
        Default is (1.0,)
    syncopate_factor : list[float]
        How syncopated the rhythm is for each pattern (0. = not syncopated at all, 1. = very syncopated).
        Default is (0.0,)
    density : list[float]
        How dense the rhythm is for each pattern, e.g. the number of events per beat. (0. = not dense, 1. = very dense)
        Default is (1.0,)
    event_var : list[float]
        The event variation for each pattern. *This is not really used now since each pattern is generated from scratch
        rather than evolving from an existing pattern*
        Default is (1.0,)
    sync_var : list[float]
        The sync variation for each pattern. *This is not really used now since each pattern is generated from scratch
        rather than evolving from an existing pattern*
        Default is (1.0,)
    threshold : list[float]
        Filters out low probability events. (0 - 1)
        Default is (0.0,)
    weight_minimum : list[float]
        The minimum weight for rescaling, making events more probable. *Not used in kin.rhythmicator*
    dynamic_range_low : list[float]
        The low end of the range to which to rescale the amplitudes of the rhythm. Default is (0.,)
    dynamic_range_high : list[float]
        The high end of the range to which to rescale the amplitudes of the rhythm. Default is (1.,)
    """

    def __init__(self,
                 ts_num,
                 ts_denom,
                 strat_level,
                 metric_factor=(1.0,),
                 syncopate_factor=(0.0,),
                 density=(1.,),
                 event_var=(1.,),
                 sync_var=(1.,),
                 threshold=(0.,),
                 weight_minimum=(0,),
                 dynamic_range_low=(0.,),
                 dynamic_range_high=(1.,)):
        self.ts_num = ts_num
        self.ts_denom = ts_denom
        self.strat_level = strat_level
        self.metric_factor = metric_factor
        self.syncopate_factor = syncopate_factor
        self.density = density
        self.event_var = event_var
        self.sync_var = sync_var
        self.threshold = threshold
        self.weight_minimum = weight_minimum
        self.dynamic_range_low = dynamic_range_low
        self.dynamic_range_high = dynamic_range_high

        self._min_amplitude = -48.
        self._tempo = 120.
        self._sample_rate = 44100.
        self._mixing_coeffs = None
        self._patterns = None
        self._audio_files = None

        self._jam = None

    def _initialize_default_lists(self, num_patterns):
        self._extend_default_value_list('self.metric_factor', num_patterns)
        self._extend_default_value_list('self.syncopate_factor', num_patterns)
        self._extend_default_value_list('self.density', num_patterns)
        self._extend_default_value_list('self.event_var', num_patterns)
        self._extend_default_value_list('self.sync_var', num_patterns)
        self._extend_default_value_list('self.threshold', num_patterns)
        self._extend_default_value_list('self.weight_minimum', num_patterns)
        self._extend_default_value_list('self.dynamic_range_low', num_patterns)
        self._extend_default_value_list('self.dynamic_range_high', num_patterns)

    @property
    def num_patterns(self):
        return len(self.patterns)

    @property
    def num_pulses_per_measure(self):
        return [len(patt['rhythm_pattern']) for patt in self.patterns]

    @property
    def num_pulses_per_beat(self):
        return [(n // self.ts_num) for n in self.num_pulses_per_measure]

    @property
    def min_amplitude(self):
        """
        The minimum amplitude of an onset in dB. Read-only. Default is -48.
        """
        return self._min_amplitude

    @property
    def tempo(self):
        """
        The tempo which should be used during synthesis. Read-only. Default is 120.
        """
        return self._tempo

    @property
    def sample_rate(self):
        """
        Sampling rate. Read-only. Default is 44100.
        """
        return self._sample_rate

    @property
    def mixing_coeffs(self):
        """
        Mixing coefficients that determine the relative level for each synthesized pattern. Read-only.
        Default is [1./num_patterns, 1./num_pattern, ... ].
        """
        if self._mixing_coeffs is None:
            return np.ones(self.num_patterns) / self.num_patterns
        else:
            return self._mixing_coeffs

    @property
    def patterns(self):
        """
        The generated patterns. Can also be based in at initialization time (e.g., if created from a JAMS file)
        Read-only. Default is None.
        """
        return self._patterns

    @property
    def audio_files(self):
        """
        The audio files used to synthesized the patterns in their respective order by pattern.
        Read-only. Default is None.
        """
        return self._audio_files

    @property
    def jam(self):
        """
        The JAMS annotations for the current pattern. Read-only. Default is None.
        """
        return self._jam

    @property
    def duration(self):
        return self.jam.file_metadata.duration

    @staticmethod
    def from_jams(jams_file_path):
        """
        Load a pattern from a JAMS file and instantiate the RhythmSynthesizer

        Parameters
        ----------
        jams_file_path : str
            Path to jams file

        Returns
        -------
        rhythm_synth : RhythmSynthesizer
        """
        jam = jams.load(jams_file_path)

        pattern1 = jam.search(pattern_index=0)[0]
        ts_num = jam.sandbox.time_signature[0]
        ts_denom = jam.sandbox.time_signature[1]
        num_patterns = len(jam.search(namespace='onset'))
        tempo = jam.search(namespace='tempo')[0].data.value[0]
        sample_rate = jam.sandbox.sample_rate
        strat_level = []
        metric_factor = []
        syncopate_factor = []
        density = []
        threshold = []
        weight_minimum = []
        dynamic_range_low = []
        dynamic_range_high = []
        min_amplitude = pattern1.sandbox.min_amplitude
        mixing_coeffs = []
        patterns = []
        audio_files = []

        for i in range(num_patterns):
            onset_ann = jam.search(pattern_index=i)[0]
            strat_level.append(onset_ann.sandbox.strat_level)
            metric_factor.append(onset_ann.sandbox.metric_factor)
            syncopate_factor.append(onset_ann.sandbox.syncopate_factor)
            density.append(onset_ann.sandbox.density)
            threshold.append(onset_ann.sandbox.threshold)
            weight_minimum.append(onset_ann.sandbox.weight_minimum)
            dynamic_range_low.append(onset_ann.sandbox.dynamic_range[0])
            dynamic_range_high.append(onset_ann.sandbox.dynamic_range[1])
            mixing_coeffs.append(onset_ann.sandbox.mixing_coeff)
            patterns.append(_dict_of_list_to_dict_of_array(onset_ann.sandbox.patterns))
            audio_files.append(onset_ann.sandbox.audio_source)

        pattern_genr = PatternGenerator(ts_num=ts_num,
                                        ts_denom=ts_denom,
                                        strat_level=strat_level,
                                        metric_factor=metric_factor,
                                        syncopate_factor=syncopate_factor,
                                        density=density,
                                        threshold=threshold,
                                        weight_minimum=weight_minimum,
                                        dynamic_range_low=dynamic_range_low,
                                        dynamic_range_high=dynamic_range_high)

        pattern_genr._min_amplitude = min_amplitude
        pattern_genr._tempo = tempo
        pattern_genr._sample_rate = sample_rate
        pattern_genr._jam = jam
        pattern_genr._mixing_coeffs = mixing_coeffs
        pattern_genr._patterns = patterns
        pattern_genr._audio_files = audio_files

        return pattern_genr

    def _extend_default_value_list(self, attribute, num_patterns):
        if len(eval(attribute)) < num_patterns:
            print('Using default `{}`'.format(attribute))
            exec '%s *= num_patterns' % attribute

    def _generate_mono_pattern(self, idx):
        """
        Create the monophonic rhythm pattern of MIDI note velocities

        Returns
        -------
        output : dict
            'beat_pattern' : np.array
                Length of self.num_pulses_per_measure, where each non-zero value denotes an onset and the weight of
                that beat.
            'beat_probability' : np.array
                Length of self.num_pulses_per_measure, where each value is the probability used to calculate if the
                beat has an onset.
            'sync_pattern' : np.array
                Length of self.num_pulses_per_measure, where each element can be 0 or 1. 1 denotes whether that beat is
                syncopated, anticipating the next beat.
            'density' : float
                The average beat_probability.
            'amp_pattern' : np.array
                Length of self.num_pulses_per_measure, where each value represents the log-scaled amplitude of the beat.
            'rhythm_pattern' : np.array
                This what we are most concerned about. It is the combination of the beat_pattern and amp_pattern. So,
                a non-zero value means there is an onset, and the value represents the log-scaled amplitude of the
                onset.
        """
        # instead of using the sequencer density parameter, the rhythmicator implementation controls density
        # by breaking it into two values which control the range_factor of the metric weight calculation and a scaling
        density_a = min(max((self.density[idx] - 0.15) / (1 - 0.15), 0.), 1.)
        density_b = min(max(self.density[idx] / 0.15, 0.), 1.)
        stratification = stratify(self.ts_num, self.ts_denom, self.strat_level[idx])
        weights = calc_weights(stratification, density_a)[0] * density_b
        num_pulses_per_measure = weights.shape[0]

        # the sequencer generates the rhythmic pattern for a measure given our parameters
        sequencer = Sequencer(weights,
                              metric_factor=self.metric_factor[idx],
                              syncopate_factor=self.syncopate_factor[idx],
                              event_var=self.event_var[idx],
                              sync_var=self.sync_var[idx],
                              weight_minimum=self.weight_minimum[idx])
        output = sequencer.create_pattern()

        # threshold filters out low probability events
        output['beat_pattern'] *= (output['beat_pattern'] > self.threshold[idx])

        # base amplitude weights are calculated independently of the other metric weights controlled by density
        amp_weights = calc_weights(stratification, 0.5)[0]
        # when it is a syncopated / anticipatory pulse, use the amplitude weight from the beat it is anticipating
        emphasis_map = (output['sync_pattern'] +
                        np.arange(num_pulses_per_measure)) % num_pulses_per_measure

        amp_idxs = np.zeros(num_pulses_per_measure, dtype=int)
        for i in range(num_pulses_per_measure):
            if random.random() > (1.0 - self.metric_factor[idx]):
                amp_idxs[i] = np.where(weights == np.percentile(weights,
                                                                random.random() * 100, interpolation='nearest'))[0][0]
            else:
                amp_idxs[i] = emphasis_map[i]
        output['amp_pattern'] = amp_weights[amp_idxs]

        # dynamic range
        output['amp_pattern'] = output['amp_pattern'] * (self.dynamic_range_high[idx] -
                                                         self.dynamic_range_low[idx]) + self.dynamic_range_low[idx]

        # Note: I did not implement the staccato / legato or syncopated release portions of the rhythmicator
        output['rhythm_pattern'] = output['amp_pattern'] * (output['beat_pattern'] > 0)

        return output

    def generate_pattern(self, num_patterns):
        """
        Create the polyphonic rhythm pattern

        Returns
        -------
        patterns : list[dict]
            List of dicts of generated pattern arrays
        """
        self._initialize_default_lists(num_patterns)

        self._patterns = []
        for i in range(num_patterns):
            while True:
                pat = self._generate_mono_pattern(i)
                # make sure that there is at least one onset!
                if np.sum(pat['rhythm_pattern']) > 0:
                    self._patterns.append(pat)
                    break

        # if a jam exists, it is now invalid
        self._jam = None

    def generate_jam(self,
                     output_jams_file=None,
                     audio_files=None,
                     mixing_coeffs=None,
                     tempo=None,
                     sample_rate=None,
                     min_amplitude=None,
                     duration_sec=None,
                     additional_onset_sandbox_info=None,
                     additional_global_sandbox_info=None):
        """
        Parameters
        ----------
        output_jams_file : str
            If not None, write jams file to `output_jams_file`.
        audio_files : list[str]
            A list of the audio_files to use for rendering the rhythm patterns. These should be ordered according to the
            pattern (e.g., if the patterns were constructed to be from low frequency to high frequency, a bass drum
            might be first in the list). If not None, overwrite self.audio_files. Default is None.
        mixing_coeffs : np.array
            The coefficients specifying the mixing levels for the patterns. If not None, overwrite self.mixing_coeffs.
            Default is None.
        tempo : float
            Tempo for rendering. If not None, overwrite self.tempo for rendering. Default is None.
        sample_rate : float
            Sampling rate of output. If not None, overwrite self.sample_rate. Default is None.
        min_amplitude : float
            The minimum amplitude to render for the softest sound above 0 in log amplitude. If not None, overwrite
            self.min_amplitude. Default is None.
        duration_sec : float
            If not None, extend the rhythm pattern to the target duration. Default is None.
        additional_onset_sandbox_info : list[dict]
            If not None, then save this additional info in the sandbox for the corresponding onset annotation.
            Default is None.
        additional_global_sandbox_info: dict
            If not None, then save this additional info in the top-level sandbox of the JAM. Default is None.

        Returns
        -------
        jam : jams.JAM

        """
        if self.patterns is None:
            raise PyrhythmicatorError('There are no patterns. Call `PatternGenerator.generate_pattern` to '
                                      'generate patterns')

        if mixing_coeffs is not None:
            self._mixing_coeffs = mixing_coeffs

        if audio_files is not None:
            self._audio_files = audio_files
            assert (len(self.audio_files) == self.num_patterns)

        if tempo is not None:
            self._tempo = tempo

        if sample_rate is not None:
            self._sample_rate = sample_rate

        if min_amplitude is not None:
            self._min_amplitude = min_amplitude

        # calculate length of a pulse in samples given tempo and sample rate for first pattern
        (pulse_length_sec,
         pulse_length_samples,
         bar_length_sec,
         bar_length_samples) = calc_metric_durations(self.strat_level[0],
                                                     self.num_pulses_per_measure[0],
                                                     self.tempo,
                                                     self.sample_rate)

        # make JAM structure
        jam = jams.JAMS()
        jam.file_metadata.duration = bar_length_samples / float(self.sample_rate)
        jam.sandbox.sample_rate = self.sample_rate
        jam.sandbox.time_signature = (self.ts_num, self.ts_denom)
        if additional_global_sandbox_info is not None:
            jam.sandbox.update(additional_global_sandbox_info)

        # write beat positions to jams
        beat_ann = jams.Annotation(namespace='beat', time=0, duration=jam.file_metadata.duration)
        beat_ann.annotation_metadata = jams.AnnotationMetadata(data_source='generative program')
        for i in range(self.ts_num):
            beat_ann.append(time=i * self.num_pulses_per_beat[0] * pulse_length_sec,
                            duration=0.0,
                            value=i)
        jam.annotations.append(beat_ann)

        # write tempo to jams
        tempo_ann = jams.Annotation(namespace='tempo', time=0, duration=jam.file_metadata.duration)
        tempo_ann.append(time=0, duration=jam.file_metadata.duration, value=self.tempo, confidence=1.0)
        jam.annotations.append(tempo_ann)

        # write onsets for each rhythm pattern
        for i in range(len(self.patterns)):
            onsets_ann = jams.Annotation(namespace='onset', time=0, duration=jam.file_metadata.duration)
            onsets_ann.annotation_metadata = jams.AnnotationMetadata(data_source='generative program')
            onsets_ann.sandbox = jams.Sandbox(pattern_index=i,
                                              audio_source=self.audio_files[i],
                                              strat_level=self.strat_level[i],
                                              metric_factor=self.metric_factor[i],
                                              syncopate_factor=self.syncopate_factor[i],
                                              density=self.density[i],
                                              threshold=self.threshold[i],
                                              weight_minimum=self.weight_minimum[i],
                                              dynamic_range=(self.dynamic_range_low[i], self.dynamic_range_high[i]),
                                              min_amplitude=self.min_amplitude,
                                              mixing_coeff=self.mixing_coeffs[i],
                                              patterns=_dict_of_array_to_dict_of_list(self.patterns[i]))
            if additional_onset_sandbox_info is not None:
                onsets_ann.sandbox.update(additional_onset_sandbox_info[i])

            # calculate length of a pulse in samples given tempo and sample rate for current pattern
            (pulse_length_sec,
             pulse_length_samples,
             bar_length_sec,
             bar_length_samples) = calc_metric_durations(self.strat_level[i],
                                                         self.num_pulses_per_measure[i],
                                                         self.tempo,
                                                         self.sample_rate)

            # log scale the amplitude signal from self.min_amplitude to 0 dB
            rhythm_pattern_linear = log_to_linear_amp(self.patterns[i]['rhythm_pattern'], (self.min_amplitude, 0.))

            # write onsets
            for j in range(self.num_pulses_per_measure[i]):
                start_idx = int(round(j * pulse_length_samples))
                if rhythm_pattern_linear[j] > 0:
                    onsets_ann.append(time=start_idx / float(self.sample_rate),
                                      value=rhythm_pattern_linear[j],
                                      duration=0)

            jam.annotations.append(onsets_ann)

        # extend and shift patterns
        if duration_sec is not None:
            if duration_sec < jam.file_metadata.duration:
                print('Argument `duration_sec` is less than 1 bar and will not be used.')
            jam = self._extend_and_shift_patterns(jam, duration_sec)

        self._jam = jam
        if output_jams_file is not None:
            jam.save(output_jams_file)

    def synthesize(self,
                   output_file,
                   **kwargs):
        """
        Synthesize the patterns from self.jam. If self.jam does not exist, pass in the arguments to `generate_jam`.

        Parameters
        ----------
        output_file : str
            The path where the rendered and mixed output signal will be written
        kwargs : additional keyword arguments
            Additional keyword arguments to pass to `pyrhythmicator.PatternGenerator.generate_jam` if self.jam is None.

        Returns
        -------
        rhythm_audio : np.array
            The mixed output_signal
        unmixed_rhythm_audio: np.array
            An MxN array where M is the number of rhythm patterns and N is the length of the measure in samples

        See Also
        --------
        pyrhythmicator.PatternGenerator.generate_jam
        """
        if self.jam is None or len(kwargs) > 0:
            self.generate_jam(**kwargs)

        onset_anns = self.jam.search(namespace='onset')

        unmixed_rhythm_audio = np.zeros([self.num_patterns, int(np.ceil(self.duration * self.sample_rate))])

        for k, onset_ann in enumerate(onset_anns):
            # load audio file
            sample, _ = librosa.load(self.audio_files[k], self.sample_rate, mono=True)
            sample_length = sample.shape[0]

            # render samples at onsets
            for j in range(len(onset_ann.data.index)):
                try:
                    start_idx = int(np.round(onset_ann.data.ix[j, 'time'].total_seconds() * self.sample_rate))
                    stop_idx = min(start_idx + sample_length, unmixed_rhythm_audio.shape[1]-1)
                    unmixed_rhythm_audio[k, start_idx:stop_idx] = onset_ann.data.ix[j, 'value'] * sample[:(stop_idx -
                                                                                                           start_idx)]
                except Exception as e:
                    import pdb; pdb.set_trace()

        rhythm_audio = np.dot(self.mixing_coeffs, unmixed_rhythm_audio)

        _write_wav(output_file, rhythm_audio, self.sample_rate)

        return rhythm_audio, unmixed_rhythm_audio

    def _extend_and_shift_patterns(self,
                                   jam,
                                   extended_duration_sec=None,
                                   duration_distribution=scipy.stats.truncnorm(-2, 2, 30, 10),
                                   shift=True,):
        """
        Repeat the measure until its at least chosen from drawing from `duration_distribution`. If `shift` is True,
        then also randomly offset the start time.

        Parameters
        ----------
        jam : jams.JAM
        extended_duration_sec : float
            The desired duration in seconds. If None, sample from duration_distribution. Default is None.
        duration_distribution : scipy.stats.rv_continuous
            Distribution from which to draw the duration from if `extended_duration_sec` is None.
            Default is scipy.stats.truncnorm(-2, 2, 30, 10)
        shift : bool
            Randomly shift the signal by an offset. Default is True.

        Returns
        -------
        jam : jams.JAM
            Updated jam
        """
        if extended_duration_sec is None:
            extended_duration_sec = duration_distribution.rvs()

        extended_duration_samples = extended_duration_sec * self.sample_rate

        (pulse_length_sec,
         pulse_length_samples,
         bar_length_sec,
         bar_length_samples) = calc_metric_durations(self.strat_level[0],
                                                     self.num_pulses_per_measure[0], self.tempo, self.sample_rate)

        repetitions = int(np.ceil(extended_duration_samples / bar_length_samples))

        # repeat
        for ann in jam.annotations:
            if ann.namespace in ['onset', 'beat']:
                _repeat_annotations(ann, repetitions)

        # shift
        if shift:
            shift_samples = np.random.randint(bar_length_samples * repetitions)

            for ann in jam.annotations:
                if ann.namespace in ['onset', 'beat']:
                    _rotate_annotations(ann, shift_samples / self.sample_rate)

        # trim
        for ann in jam.annotations:
            if ann.namespace in ['onset', 'beat']:
                _trim_annotations(ann, 0, extended_duration_sec)
        jam.search(namespace='tempo')[0].duration = extended_duration_sec

        jam.file_metadata.duration = extended_duration_sec
        return jam
