# minimed-rag-kg environment restore

This repository was reconstructed from the remaining `.venv`.

Python version in the existing environment:

```bash
Python 3.11.15
```

Recreate the virtual environment with `uv`:

```bash
uv python install 3.11.15
uv venv --python 3.11.15 .venv
uv pip install -r requirements.txt -r requirements-dev.txt
```

Or with stdlib `venv` and `pip`:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
```

Notes:

- `requirements.txt` was reconstructed from `.venv/lib/python3.11/site-packages/*.dist-info/METADATA`.
- The previous source tree was deleted, so this restores dependency metadata only.
- The previous editable/local package `minimed-rag-kg==0.1.0` existed in `.venv`, but the source package files are no longer present.
