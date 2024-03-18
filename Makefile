install:
	pip install --upgrade pip && pip install -r requirements.txt
format:
	find src -name '*.py' -exec black {} +
lint:
	pylint --disable=R,C src/*.py \
	return 0
test:
	python -m pytest