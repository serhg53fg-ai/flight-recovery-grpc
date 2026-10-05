import pytest
from training.baselines import ScheduleZeroBaseline
from tests.training.test_evaluate import labels, predictions


def test_gate_compares_identical_subset_and_counts_failures():
    from training.candidate import compare_candidate
    truth = labels()
    for item in truth:
        item['features'] = {'departure_airport': 'ZGGG'}
    result = compare_candidate(truth, predictions()[:1], predictions(), ['b'], 'same', 'same')
    assert result['candidate']['structured_success_rate'] == .5
    assert result['comparison_sample_count'] == 1
    assert result['gate']['passed'] is False
    assert 'structured_success_below_99_percent' in result['gate']['failures']
    with pytest.raises(ValueError):
        compare_candidate(truth, predictions()[:1], predictions(), [], 'same', 'same')


def test_all_failed_candidate_is_explicit_refusal():
    from training.candidate import compare_candidate
    result = compare_candidate(labels(), [], predictions(), ['a', 'b'], 'same', 'same')
    assert result['gate']['passed'] is False
    assert result['candidate']['structured_success_rate'] == 0
    assert result['candidate']['absolute_times'] is None
    assert result['comparison_sample_count'] == 0
