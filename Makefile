.PHONY: install test attack run-app lint

PY ?= python

install:  ## install dependencies
	$(PY) -m pip install -r requirements.txt

test:  ## run engine + vision unit tests
	$(PY) -m unittest discover -s autosettle_engine/tests -t .
	$(PY) -m unittest discover -s autosettle_vision/tests -t .

attack:  ## run the engine exploit / attack suite
	$(PY) -m autosettle_engine.exploit_suite

run-app:  ## run the web app without Qwen (deterministic template mode)
	$(PY) -m autosettle_vision.app --no-qwen

lint:  ## basic syntax check of all project Python files
	$(PY) -m py_compile autosettle_engine/*.py autosettle_engine/tests/*.py \
	                 autosettle_vision/*.py autosettle_vision/tests/*.py tools/*.py scripts/*.py
