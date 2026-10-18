# Setup environment and run tests

### Install `uv` (Linux and macOS)

If you do not have `uv` installed:

```bash
# Using standalone installer:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or on macOS via Homebrew:
brew install uv
```

---

### Using `uv` (Recommended)

1. Setup virtual environment:
   ```bash
   uv venv --python 3.12 .venv
   ```

2. Activate virtual environment:
   ```bash
   source .venv/bin/activate
   ```

3. Install requirements:
   ```bash
   uv pip install -r samples/requirements.txt
   ```

4. Run tests:
   ```bash
   python -m unittest samples/vit/vit_test.py
   ```

*(Alternatively, run directly with `uv run` without manual activation):*
```bash
uv run python -m unittest samples/vit/vit_test.py
```

---

### Using standard `python3` (requires `python3-venv`)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r samples/requirements.txt
python -m unittest samples/vit/vit_test.py
```
