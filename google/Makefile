include ~/nk.mk

## Extract maps-related files
extract-maps:
	python3 main.py extract_re 'Maps|[pP]lace|Saved'

## Extract tasks-related files
extract-tasks:
	python3 main.py extract_re '[tT]ask'

## Read the GPS timeline
timeline:
	python3 timeline.py backups/Timeline.json
