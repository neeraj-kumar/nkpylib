# makes a lot of make commands much easier
SHELL := /bin/bash

main: help

## print help for this Makefile
help:
	@awk '/^[a-zA-Z\-\_0-9]+:/ { \
		helpMessage = match(lastLine, /^## (.*)/); \
		if (helpMessage) { \
			helpCommand = substr($$1, 0, index($$1, ":")-1); \
			helpMessage = substr(lastLine, RSTART + 3, RLENGTH); \
			printf "\033[36m%-20s\033[0m %s\n", helpCommand, helpMessage; \
		} \
	} \
	{ lastLine = $$0 }' $(MAKEFILE_LIST)

BASE_PY_FILES=graphutils.py image_features.py
PY_FILES=$(BASE_PY_FILES)

## runs various code cleanup/checking: black, autoimport, mypy and pylint
lint:
	#autoimport $(PY_FILES)
	isort --atomic -q $(PY_FILES)
	python -m black --line-length=100 -q $(PY_FILES)
	#pylint $(PY_FILES)
	mypy $(BASE_PY_FILES)

## tests out the file watcher
watcher:
	python3 file_watcher.py ~/dp/Recipes/ ./
