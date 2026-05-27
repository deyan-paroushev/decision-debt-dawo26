.PHONY: reproduce test clean

reproduce:
	python scripts/reproduce_section_5_2.py

test:
	python -m unittest discover -s tests

clean:
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
