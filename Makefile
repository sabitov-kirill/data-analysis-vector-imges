all:
	python3 src/voronoi.py
	python3 src/cross-validation.py
	python3 src/cross-validation.py --rows=4 --slice_rows --out=out/cross-validation-with-time.svg