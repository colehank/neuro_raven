# QuickUse
## Installation
Code is tested with **Python 3.11+** on Linux and MacOS.

```bash
git clone https://github.com/colehank/neuro_raven.git
cd neuro_raven
```

This project uses [`uv`]((https://docs.astral.sh/uv/getting-started/installation/)) for Python dependency management.

if you have no `uv` installed, you can install it with the following command:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

and then sync the dependencies:
```bash
uv sync
```

after this, you should be able to run all analyses in this project.