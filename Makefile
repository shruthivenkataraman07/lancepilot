PY ?= .venv/bin/python
export PYTHONPATH := src

.PHONY: install test lint typecheck benchmark demo clean

install:
	python3.12 -m venv .venv
	$(PY) -m pip install -e '.[dev]'

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .

typecheck:
	$(PY) -m mypy src

benchmark:
	$(PY) -m lancepilot.cli benchmark --config configs/phase_shift.yaml configs/mixed_size.yaml
	$(PY) -m lancepilot.cli report --results results

demo:
	@echo "Measured locally on synthetic workload"
	$(PY) -m lancepilot.cli benchmark --config configs/phase_shift.yaml
	@echo "chart: results/phase_shift_rolling_hit_ratio.png"

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache
