include ~/nk.mk

BASE_PY_FILES=collections.py model.py workers.py
OTHER_PY_FILES=
PY_FILES=$(BASE_PY_FILES)


## run nk collections web server
serve:
	python3 nkcollections.py db/tumblr/{collection.sqlite,embeddings.lmdb}
