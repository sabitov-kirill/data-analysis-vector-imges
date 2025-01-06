VENV := .venv
PYTHON := $(VENV)/bin/python3
PIP := $(VENV)/bin/pip

all: voronoi crossvalidation kd-tree hat-loss bayes

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt

voronoi: $(VENV)/bin/activate
	$(PYTHON) src/voronoi.py

crossvalidation: $(VENV)/bin/activate
	$(PYTHON) src/cross-validation.py
	$(PYTHON) src/cross-validation.py --rows=4 --slice_rows --out=out/cross-validation-with-time.svg

kd-tree: $(VENV)/bin/activate
	$(PYTHON) src/kd-tree.py

hat-loss: $(VENV)/bin/activate
	$(PYTHON) src/hat-loss.py

bayes: $(VENV)/bin/activate
	$(PYTHON) src/bayes-dividing-surface.py save --out=out/bayes-circular-close.svg --mean1 0 0 --mean2 2 2 --cov1 1 0 0 1 --cov2 1 0 0 1
	$(PYTHON) src/bayes-dividing-surface.py save --out=out/bayes-default.svg
	$(PYTHON) src/bayes-dividing-surface.py save --out=out/bayes-rotated.svg --mean1 0 0 --mean2 2 2 --cov1 1 0 0 1 --cov2 2 -1.5 -1.5 2
	$(PYTHON) src/bayes-dividing-surface.py save --out=out/bayes-nested.svg --mean1 0 0 --mean2 0 0 --cov1 1 0 0 1 --cov2 2 0 0 2 --xlim -2 2 --ylim -2 2
	$(PYTHON) src/bayes-dividing-surface.py save --out=out/bayes-crossed.svg --mean1 0 0 --mean2 0 0 --cov1 2 1 1 1 --cov2 1 -1 -1 2 --xlim -2 2 --ylim -2 2
	$(PYTHON) src/bayes-dividing-surface.py save --out=out/bayes-contained.svg --mean1 0 0 --mean2 0 0 --cov1 2 0 0 2 --cov2 2 1 1 1 --xlim -2 2 --ylim -2 2 --eye 1.5 -2 0.25

clean:
	rm -rf $(VENV)