# tests/unit/test_metrics.py
import math
import pytest

from src.metrics import latency_stats, hit_rate, throughput, correctness


def test_latency_stats_basic():
    samples = [0.1, 0.2, 0.3, 0.4, 0.5]
    stats = latency_stats(samples)
    assert math.isclose(stats["mean"], 0.3, rel_tol=1e-6)
    # p95 should be near the top of the distribution
    assert 0.4 <= stats["p95"] <= 0.5
    # p99 should equal the maximum for such a small set
    assert stats["p99"] == 0.5


def test_latency_stats_empty():
    stats = latency_stats([])
    assert stats["mean"] == 0.0
    assert stats["p95"] == 0.0
    assert stats["p99"] == 0.0


def test_hit_rate():
    assert hit_rate(5, 10) == 0.5
    assert hit_rate(0, 0) == 0.0
    assert hit_rate(3, 3) == 1.0


def test_throughput():
    assert throughput(10, 2.0) == 5.0
    assert throughput(10, 0.0) == 0.0


def test_correctness_numeric_exact():
    assert correctness(["42"], "Answer: 42")
    assert correctness(["42"], "The result is 42")
    assert correctness(["42.0"], "42")


def test_correctness_numeric_tolerance():
    # within floating point tolerance
    assert correctness(["3.1415927"], "Answer: 3.141593")


def test_correctness_numeric_fail():
    assert not correctness(["100"], "Answer: 42")


def test_correctness_textual_match():
    assert correctness(["Paris"], "The capital is Paris.")
    assert correctness(["the capital is paris"], "Paris")
    assert correctness(["Paris"], "paris")  # case insensitive


def test_correctness_textual_fail():
    assert not correctness(["London"], "The capital is Paris.")
