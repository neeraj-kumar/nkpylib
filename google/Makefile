include ~/nk.mk

## Extract maps-related files
extract-maps:
	python3 main.py extract_re 'Maps|[pP]lace|Saved'

## Extract tasks-related files
extract-tasks:
	python3 main.py extract_re '[tT]ask'

## Extract calendar-related files
extract-cal:
	python3 main.py extract_re '[Cc]alendar'

## Read the GPS timeline
timeline:
	python3 timeline.py backups/Timeline.json

## read calendar events
cal:
	python3 gcalendar.py
	-#python3 gcalendar.py "backups/Takeout/Calendar/NKAK Health.ics"
