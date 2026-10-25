import numpy as np
import pytest

from arbor.criteria import (
    conditional_entropy,
    entropy,
    entropy_rows,
    gain_ratio,
    information_gain,
    split_information,
)


def test_entropy_of_play_tennis_target():
    # 9 yes, 5 no: the value every textbook quotes.
    assert entropy([9, 5]) == pytest.approx(0.940, abs=1e-3)


def test_entropy_bounds():
    assert entropy([10, 0]) == 0.0
    assert entropy([5, 5]) == pytest.approx(1.0)
    assert entropy([1, 1, 1, 1]) == pytest.approx(2.0)
    assert entropy([0, 0]) == 0.0


def test_entropy_accepts_fractional_weights():
    assert entropy([4.5, 4.5]) == pytest.approx(1.0)


def test_entropy_rows_matches_scalar_version():
    table = np.array([[2, 3], [4, 0], [3, 2]])
    assert entropy_rows(table) == pytest.approx([entropy(row) for row in table])


def test_information_gain_of_outlook():
    # Outlook splits play tennis into sunny (2 yes, 3 no), overcast (4, 0), rain (3, 2).
    partitions = np.array([[2, 3], [4, 0], [3, 2]])
    assert information_gain(np.array([9, 5]), partitions) == pytest.approx(0.2467, abs=1e-4)
    assert conditional_entropy(partitions) == pytest.approx(0.6935, abs=1e-4)


def test_split_information_and_gain_ratio_of_outlook():
    split_info = split_information([5, 4, 5])
    assert split_info == pytest.approx(1.5774, abs=1e-4)
    assert gain_ratio(0.2467, split_info) == pytest.approx(0.1564, abs=1e-4)


def test_gain_ratio_is_zero_without_split_information():
    assert gain_ratio(0.5, 0.0) == 0.0
