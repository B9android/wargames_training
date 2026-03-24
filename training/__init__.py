"""Wargames Training — training public API (E12.2).

Stable interfaces for training runners, evaluation utilities, and benchmark
tools.  Import from this module to remain insulated from internal restructuring.

Training runners
----------------
:func:`~training.train.train` — train a single-agent policy with PPO.
:func:`~training.self_play.evaluate_vs_pool` — evaluate against an opponent pool.

Evaluation
----------
:func:`~training.evaluate.evaluate` — quick win-rate evaluation.
:func:`~training.evaluate.run_episodes_with_model` — detailed episode runner.

Benchmarks
----------
:class:`~training.wfm1_benchmark.WFM1Benchmark` — WFM-1 zero-shot evaluation.
:class:`~training.transfer_benchmark.TransferBenchmark` — GIS transfer benchmark.
:class:`~training.historical_benchmark.HistoricalBenchmark` — historical fidelity.
"""
