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

BASE_PY_FILES=chroma.py airtable.py recipes.py web_utils.py web_search.py constants.py state_logger.py thread_utils.py geo.py tasks.py fs_tree.py cli_utils.py memento.py our_groceries.py
# utils.py
PY_FILES=$(BASE_PY_FILES)

## runs typechecking on various files
mypy:
	mypy $(PY_FILES)

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

## geocoder tests
geo:
	python3 geo.py


## test web searcher
search:
	#python3 web_search.py parrot
	#python3 web_search.py "lenin imperialism highest stage of capitalism"
	#python3 web_search.py "thorf - dagrenning"
	#python3 web_search.py "1.2lb in g"
	#python3 web_search.py "rice wine vinegar substitute"
	#python3 web_search.py "site:wikipedia.org philosophy"
	#python3 web_search.py "asoid"
	python3 web_search.py "site:www.the-pasta-project.com busiate pasta with trapani pesto"

## run the CLI utility
cli:
	python3 cli_utils.py

## tests out the OurGroceries API
og:
	python3 our_groceries.py

## tests out the Memento database integration
memento:
	python3 memento.py
