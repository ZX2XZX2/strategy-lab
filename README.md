# strategy-lab

## Overview
Strategy-Lab is a modular backtesting and trading framework for developing and evaluating stock trading strategies using EOD and intraday data.

## Installation

1. Install Poetry:
```bash
curl -sSL https://install.python-poetry.org | python3 -
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

2. Install project dependencies:
```bash
cd strategy-lab
poetry install
```

3. Activate the virtual environment:
```bash
poetry env activate
```

## Running Tests

Simply run:
```bash
pytest
```

## Using VSCode with Poetry

1. Open the project folder in VSCode:
```bash
code .
```

2. Press `Ctrl+Shift+P` → "Python: Select Interpreter"

3. Choose the Poetry virtualenv interpreter (you can find it with `poetry env info --path`)

4. VSCode will now use the correct environment for linting, running, and testing.

## Project Structure

- `strategy_lab/data/` — Data loading and adjustments
- `strategy_lab/selection/` — Stock selectors
- `strategy_lab/execution/` — Trading logic
- `strategy_lab/backtest/` — Backtest runner and metrics
- `strategy_lab/utils/` — Utility functions
- `scripts/` — Data preparation scripts
- `tests/` — Unit tests
