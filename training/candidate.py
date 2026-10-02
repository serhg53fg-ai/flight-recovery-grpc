"""Recompute candidate and reference metrics from record-level predictions."""
from dataclasses import asdict
from .evaluate import evaluate_predictions, check_release_gate


def compare_candidate(labels, predictions, reference, failed_ids, label_hash, prediction_hash):
    options = dict(label_manifest_hash=label_hash, prediction_label_hash=prediction_hash,
                   group_fields=('departure_airport', 'arrival_airport'), minimum_group_size=20)
    if not predictions:
        if (not label_hash or label_hash != prediction_hash or not labels or
                len(failed_ids) != len(set(failed_ids)) or
                len(labels) != len({row['record_id'] for row in labels}) or
                set(failed_ids) != {row['record_id'] for row in labels}):
            raise ValueError('invalid all-failure coverage or hash')
        evaluate_predictions(labels, reference, **options)
        return {'dataset_manifest_hash': label_hash,
                'candidate': {'sample_count': len(labels), 'structured_success_rate': 0,
                              'absolute_times': None, 'components': None, 'time_order_rate': None},
                'reference_on_candidate_successes': None, 'comparison_sample_count': 0,
                'gate_metrics': None, 'gate': {'passed': False, 'failures': ['no_successful_predictions']}}
    candidate = evaluate_predictions(labels, predictions, failed_record_ids=failed_ids, **options)
    full_reference = evaluate_predictions(labels, reference, **options)
    success_ids = {row['record_id'] for row in predictions}
    subset_labels = [row for row in labels if row['record_id'] in success_ids]
    matched_reference = evaluate_predictions(subset_labels,
        [row for row in reference if row['record_id'] in success_ids], **options)
    regression = 0.0
    for key in full_reference.groups:
        # A required group disappearing through failures must not silently pass.
        if key not in candidate.groups or key not in matched_reference.groups:
            regression = float('inf')
            break
        baseline = matched_reference.groups[key]['arrival_mae']
        actual = candidate.groups[key]['arrival_mae']
        regression = max(regression, (actual - baseline) / baseline if baseline else
                         (0.0 if actual == 0 else float('inf')))
    values = dict(sample_count=len(labels), departure_mae=candidate.absolute_times['off_block']['mae'],
                  arrival_mae=candidate.absolute_times['on_block']['mae'],
                  structured_success_rate=candidate.structured_success_rate,
                  time_order_rate=candidate.time_order_rate, maximum_subgroup_regression=regression)
    schedule = dict(departure_mae=matched_reference.absolute_times['off_block']['mae'],
                    arrival_mae=matched_reference.absolute_times['on_block']['mae'])
    if regression == float('inf'):
        # Keep JSON finite; a missing required group is an explicit gate failure.
        values['maximum_subgroup_regression'] = 1.0
    gate = check_release_gate(values, schedule)
    return {'dataset_manifest_hash': label_hash, 'candidate': asdict(candidate),
            'reference_on_candidate_successes': asdict(matched_reference),
            'comparison_sample_count': len(subset_labels), 'gate_metrics': values, 'gate': asdict(gate)}
