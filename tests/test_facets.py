import pytest

from ccpf.answer.aggregate import compute_component_stats
from ccpf.answer.facets import duration_window, parse_duration_months


@pytest.mark.parametrize("text,expected", [
    ("builder delayed possession by 3 years", 36),
    ("delay of 18 months", 18),
    ("it has been two years since", 24),
    ("1.5 years late", 18),
    ("booked a flat in 2015, seeking compensation", None),
    ("paid 23 lac", None),
])
def test_parse_duration_months(text, expected):
    assert parse_duration_months(text) == expected


def test_duration_window_is_proportional_with_floor():
    assert duration_window(36) == (22, 50)
    assert duration_window(6) == (0, 12)  # min slack of 6 months


def test_quartiles_and_values_exposed_for_distribution():
    rows = [{"tid": i, "judgment": {"x": float(v)}} for i, v in enumerate([10, 20, 30, 40, 50])]
    s = compute_component_stats("x", rows, lambda j: j["x"], sample_size=5)
    assert (s.p25, s.median, s.p75) == (20, 30, 40)
    assert s.values == [10, 20, 30, 40, 50]


def test_quartiles_omitted_for_tiny_samples():
    rows = [{"tid": i, "judgment": {"x": float(v)}} for i, v in enumerate([1, 2, 3])]
    s = compute_component_stats("x", rows, lambda j: j["x"], sample_size=3)
    assert s.p25 is None and s.p75 is None
