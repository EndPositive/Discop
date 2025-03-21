# Discop UDP server

First, please ensure that you have installed all the required libraries for this repository.

```shell
uv venv
source .venv/bin/activate
uv sync
python src/setup.py build_ext --build-lib=src/
```

### Run Single Example

You can modify the default settings for each generation task in `src/config.py`.

```shell
pytest src/stega_cy_test.py
```
