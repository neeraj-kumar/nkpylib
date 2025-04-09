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

BASE_PY_FILES=faces.py features.py embeddings.py llm_utils.py bpm.py server.py
OTHER_PY_FILES=client.py constants.py replicate_wrapper.py providers.py text.py
PY_FILES=$(BASE_PY_FILES)

## runs various code cleanup/checking: black, autoimport, mypy and pylint
lint: mypy
	#autoimport $(PY_FILES)
	#isort --atomic -q $(PY_FILES)
	#isort --atomic --om --df -c ~/isort.cfg $(PY_FILES)
	#isort --atomic -c ~/isort.cfg $(PY_FILES)
	#python -m black --line-length=100 -q $(PY_FILES)
	#pylint $(PY_FILES)

## runs mypy on files
mypy:
	mypy --explicit-package-bases $(BASE_PY_FILES)
	mypy $(OTHER_PY_FILES)

## runs the llm sample code
llm:
	python3 server.py

## runs the llm server
llm-server:
	uvicorn server:app --reload --port 8234 --host 0.0.0.0

## test requests to the llm server
test-llm:
	curl -s -X POST -H "Content-Type: application/json" --data '{"prompt": "what is the capital of france? one word only then stop", "max_tokens": 2}' http://localhost:8234/v1/completions | json
	curl -s -X POST -H "Content-Type: application/json" --data '{"input": "what is the capital of france? one word only then stop"}' http://localhost:8234/v1/embeddings
	curl -s -X POST -H "Content-Type: application/json" --data '{"input": "what is the capital of france? one word only then stop","model":"clip"}' http://localhost:8234/v1/embeddings
	curl -s -X POST -H "Content-Type: application/json" --data '{"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Ichthyotitan_Size_Comparison.svg/512px-Ichthyotitan_Size_Comparison.svg.png"}' http://localhost:8234/v1/image_embeddings
	curl -s -X POST -H "Content-Type: application/json" --data '{"url": "512px-Ichthyotitan_Size_Comparison.svg.png"}' http://localhost:8234/v1/image_embeddings
	curl -s -X POST -H "Content-Type: application/json" --data '{"a": "dog", "b": "cat"}' http://localhost:8234/v1/strsim
	curl -s -X POST -H "Content-Type: application/json" --data '{"a": "dog", "b": "rocket"}' http://localhost:8234/v1/strsim
	curl -s -X POST -H "Content-Type: application/json" --data '{"a": "dog", "b": "embedding"}' http://localhost:8234/v1/strsim

## tests the client
test-client:
	python3 client.py

## test llama
test-llama:
	curl -s -X POST -H "Content-Type: application/json" --data '{"prompt": "output the 10 most populous countries of the world and their populations as a JSON list.", "max_tokens": 500, "model": "llama3"}' http://localhost:8234/v1/completions

## test out face detection using replicate
rep-face:
	python3 replicate_wrapper.py face_detection

## test out llm completion using replicate
rep-llm:
	@python3 replicate_wrapper.py llm

## test out retina face
face-retina:
	@python3 faces.py -d retina radiohead.jpg game.jpg

## test out replicate face detection
face-replicate:
	@python3 faces.py -d replicate radiohead.jpg game.jpg

## update providers
update-providers:
	@python3 providers.py update_providers

## tests out embeddings main code
embeddings:
	@python3 embeddings.py ~/db/nkmovies/embeddings/movie-scalars.lmdb
