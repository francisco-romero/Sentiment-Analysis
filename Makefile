install:
    pip install --upgrade pip&&\
    pip install -r requirements.txt
format:
    find src -name '*.py' -exec black {} +
lint:
    pylint --disable=R,C src/*.py
test:
    python -m pytest