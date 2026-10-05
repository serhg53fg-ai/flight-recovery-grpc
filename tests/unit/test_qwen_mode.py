import pytest


def test_base_output_mode_is_explicit_and_adapter_conflicts_fail():
    from services.inference.adapters.qwen import resolve_output_mode
    assert resolve_output_mode(None, False) == 'absolute_times'
    assert resolve_output_mode('duration_components', False) == 'duration_components'
    assert resolve_output_mode(None, True) == 'duration_components'
    with pytest.raises(ValueError): resolve_output_mode('absolute_times', True)
    with pytest.raises(ValueError): resolve_output_mode('auto', False)


def test_worker_mode_flag_reaches_server():
    from deploy.autodl.worker import parse_args, build_server_argv
    args = parse_args(['--worker-id','mode-worker','--backend','qwen','--model-path','/tmp/model',
                      '--output-mode','duration_components'])
    argv = build_server_argv(args)
    assert argv[argv.index('--output-mode') + 1] == 'duration_components'
