# Linux / macOS / WSL / CI entry points. On Windows use `python tasks.py <target>`.
PY ?= python

.PHONY: help setup seed ingest serve test lint fmt typecheck eval bench openapi clean check docker-up docker-down

help:
	@$(PY) tasks.py --list

setup:          ; $(PY) tasks.py setup
seed:           ; $(PY) tasks.py seed
ingest:         ; $(PY) tasks.py ingest
serve:          ; $(PY) tasks.py serve
test:           ; $(PY) tasks.py test
lint:           ; $(PY) tasks.py lint
fmt:            ; $(PY) tasks.py fmt
typecheck:      ; $(PY) tasks.py typecheck
eval:           ; $(PY) tasks.py eval
bench:          ; $(PY) tasks.py bench
openapi:        ; $(PY) tasks.py openapi
clean:          ; $(PY) tasks.py clean
check:          ; $(PY) tasks.py check

docker-up:      ; docker compose up --build
docker-down:    ; docker compose down -v
