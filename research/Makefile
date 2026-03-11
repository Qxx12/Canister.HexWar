.PHONY: test lint fmt install clean

# Run all tests
test:
	python3 -m pytest tests/ -v

# Run tests with coverage
test-cov:
	python3 -m pytest tests/ --cov=hexwar --cov-report=term-missing

# Lint
lint:
	python3 -m ruff check hexwar/ tests/ scripts/

# Format
fmt:
	python3 -m ruff format hexwar/ tests/ scripts/

# Install in editable mode with all dependencies
install:
	pip install -e ".[dev]"

# Train CMA-ES agent (quick smoke test: 5 generations)
train-evo-smoke:
	python3 scripts/train_cmaes.py --generations 5 --games 5 --popsize 4 --output /tmp/cmaes_smoke

# Train PPO agent (quick smoke test: 3 iterations)
train-ppo-smoke:
	python3 scripts/train_ppo.py --iterations 3 --episodes 2 --max-turns 50 --output /tmp/ppo_smoke

# Run a quick tournament
tournament:
	python3 scripts/run_tournament.py --games 50 --max-turns 200

# Clean generated artifacts
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov/
	rm -rf runs/ /tmp/cmaes_smoke /tmp/ppo_smoke
