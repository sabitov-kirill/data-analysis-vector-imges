VENV := .venv
PYTHON := $(VENV)/bin/python3
PIP := $(VENV)/bin/pip
OUT := out

all: voronoi crossvalidation kd-tree regularizations hat-loss bayes defision-tree-depth pca tsne drawio

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt

$(OUT):
	mkdir -p $(OUT)

voronoi: $(VENV)/bin/activate $(OUT)
	$(PYTHON) src/voronoi.py

crossvalidation: $(VENV)/bin/activate $(OUT)
	$(PYTHON) src/cross-validation.py
	$(PYTHON) src/cross-validation.py --rows=4 --slice_rows --out=$(OUT)/cross-validation-with-time.svg

kd-tree: $(VENV)/bin/activate $(OUT)
	$(PYTHON) src/kd-tree.py

regularizations: $(VENV)/bin/activate $(OUT)
	$(PYTHON) src/regularizations.py

hat-loss: $(VENV)/bin/activate $(OUT)
	$(PYTHON) src/hat-loss.py

bayes: $(VENV)/bin/activate $(OUT)
	$(PYTHON) src/bayes-dividing-surface.py save --out=$(OUT)/bayes-circular-close.png --mean1 0 0 --mean2 2 2 --cov1 1 0 0 1 --cov2 1 0 0 1
	$(PYTHON) src/bayes-dividing-surface.py save --out=$(OUT)/bayes-default.png
	$(PYTHON) src/bayes-dividing-surface.py save --out=$(OUT)/bayes-rotated.png --mean1 0 0 --mean2 2 2 --cov1 1 0 0 1 --cov2 2 -1.5 -1.5 2
	$(PYTHON) src/bayes-dividing-surface.py save --out=$(OUT)/bayes-nested.png --mean1 0 0 --mean2 0 0 --cov1 1 0 0 1 --cov2 2 0 0 2 --xlim -2 2 --ylim -2 2
	$(PYTHON) src/bayes-dividing-surface.py save --out=$(OUT)/bayes-crossed.png --mean1 0 0 --mean2 0 0 --cov1 2 1 1 1 --cov2 1 -1 -1 2 --xlim -2 2 --ylim -2 2
	$(PYTHON) src/bayes-dividing-surface.py save --out=$(OUT)/bayes-contained.png --mean1 0 0 --mean2 0 0 --cov1 2 0 0 2 --cov2 2 1 1 1 --xlim -2 2 --ylim -2 2 --eye 1.5 -2 0.25

defision-tree-depth: $(VENV)/bin/activate $(OUT)
	$(PYTHON) src/decision-tree-depth.py

pca: $(VENV)/bin/activate $(OUT)
	$(PYTHON) src/pca.py

tsne: $(VENV)/bin/activate $(OUT)
	$(PYTHON) src/tsne.py

drawio: diagrams/
	drawio -r --crop --export --format svg --output out/ diagrams/

clean:
	rm -rf $(VENV)
	rm -rf $(OUT)

.PHONY: all clean out